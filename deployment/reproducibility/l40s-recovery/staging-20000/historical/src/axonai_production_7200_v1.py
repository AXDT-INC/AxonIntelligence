#!/usr/bin/env python3
"""
AxonAI +7,200 production generator/verifier v1.

Checkpointable, resumable, source-frozen production run.

Frozen contract
---------------
Generator: Qwen/Qwen3.8-27B
Verifier:  Qwen/Qwen3-32B-V3-two-stage
Input:     final-generation-source-manifest-7200.jsonl
Target:    exactly 7,200 accepted canonical SFT records

Safety / reproducibility
------------------------
- Never substitutes a source unit.
- Never mutates the frozen input manifest.
- Saves state after every slot in SQLite.
- Writes append-only per-attempt audit JSONL.
- Can be stopped with Ctrl+C and resumed with the same command.
- Loads only one large model at a time.
- Retries rejected slots with verifier feedback; deterministic prompts.
- Stops non-zero if any slot exhausts the configured attempt budget.
- Final canonical JSONL is rebuilt from accepted SQLite state.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sqlite3
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


VERSION = "axonai-7200-production-v1"
GENERATOR_NAME = "Qwen/Qwen3.8-27B"
VERIFIER_NAME = "Qwen/Qwen3-32B-V3-two-stage"

AXONAI_SYSTEM = (
    "You are AxonAI, a rigorous scientific research assistant specializing in "
    "biomedical science, bioinformatics, scientific computing, research "
    "methodology, and evidence-based reasoning. Distinguish observation from "
    "inference, preserve uncertainty, avoid unsupported claims, and reason "
    "carefully from available scientific evidence."
)

REQUIRED_TOP_KEYS = {
    "id",
    "messages",
    "domain",
    "task_type",
    "source",
    "license",
    "quality_score",
    "provenance",
    "verification",
}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS slots (
    slot_id TEXT PRIMARY KEY,
    ordinal INTEGER NOT NULL,
    source_json TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'PENDING',
    candidate_json TEXT,
    grounding_json TEXT,
    quality_json TEXT,
    feedback_json TEXT,
    canonical_json TEXT,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_slots_status ON slots(status);
CREATE INDEX IF NOT EXISTS idx_slots_ordinal ON slots(ordinal);
"""


def req(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def now():
    return time.time()


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            req(isinstance(obj, dict), f"{path}:{n}: expected JSON object")
            rows.append(obj)
    return rows


def append_jsonl(path: Path, row):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"no JSON object found: {text[:500]!r}")
    return json.loads(text[start:end + 1])


def load_model(path: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        str(path),
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def unload_model(tokenizer, model):
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)


def generate_json(tokenizer, model, messages, max_new_tokens):
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(rendered, return_tensors="pt")
    input_device = model.get_input_embeddings().weight.device
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = out[0, inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    return extract_json(raw), raw


def source_excerpt(slot, limit=18000):
    return (slot.get("text") or "")[:limit]


def generator_prompt(slot, attempt, feedback=None, previous_candidate=None):
    meta = {
        "slot_id": slot["slot_id"],
        "domain": slot["domain"],
        "task_type": slot["task_type"],
        "source_family": slot["source_family"],
        "source": slot.get("source"),
        "source_file": slot.get("source_file"),
        "section_title": slot.get("section_title"),
        "source_sections": slot.get("source_sections"),
        "primary_stratum": slot.get("primary_stratum"),
        "domain_hint": slot.get("domain_hint"),
        "attempt": attempt,
    }

    retry = ""
    if feedback:
        retry = f"""
This is a correction attempt. The previous candidate was rejected.

Verifier feedback:
{json.dumps(feedback, ensure_ascii=False)}

Previous candidate:
{json.dumps(previous_candidate, ensure_ascii=False)}

Correct the demonstrated problem. Do not repeat unsupported or poorly routed content.
"""

    return f"""Create exactly ONE supervised-fine-tuning example grounded only in the supplied source.

Target domain: {slot["domain"]}
Target task type: {slot["task_type"]}

The user question must genuinely require the assigned task type.
The assistant answer must be fully supported by the source.
Every substantive sentence in the answer must be directly entailed by the supplied source.
Do not add ancillary facts merely because they are plausible, related, or mentioned elsewhere.
Do not strengthen a local step, example, or tutorial action into the workflow's overall objective unless the source explicitly says so.
If the source supports only a narrower claim, state only that narrower claim.
Do not invent facts, versions, results, commands, mechanisms, recommendations, purposes, or scope.
Preserve uncertainty and limitations present in the source.
Answer only what the question asks; omit extra "Additionally..." details unless they are necessary and directly supported.
Do not mention that this is a training example or quote the task label.
Avoid trivial copy-paste questions.
For code/technical sources, reason from the supplied implementation/documentation.
For scientific literature, distinguish findings from interpretation.
{retry}
Return ONLY:
{{
  "question": "<clear user question>",
  "answer": "<rigorous source-grounded answer>"
}}

Source metadata:
{json.dumps(meta, ensure_ascii=False)}

Source:
---BEGIN SOURCE---
{source_excerpt(slot)}
---END SOURCE---
"""


def validate_candidate(obj):
    req(isinstance(obj, dict), "candidate is not object")
    req(isinstance(obj.get("question"), str), "question missing")
    req(isinstance(obj.get("answer"), str), "answer missing")
    req(len(obj["question"].strip()) >= 12, "question too short")
    req(len(obj["answer"].strip()) >= 40, "answer too short")


def grounding_prompt(slot, qa):
    return f"""Audit this candidate SFT example ONLY against the supplied source.

Return ONLY:
{{
  "fully_grounded": true,
  "unsupported_claims": [],
  "grounding_rationale": "<concise rationale>"
}}

Rules:
- fully_grounded must be false if the answer contains any material claim not supported by the source.
- unsupported_claims must contain short descriptions of unsupported claims.
- Do not reward plausibility; require direct source support.
- Treat statements about an overall workflow objective, purpose, scope, or intended outcome as unsupported unless the source explicitly states them.
- A local action or tutorial step must not be generalized into the primary objective of the whole workflow without explicit support.
- Commands, versions, numeric values, mechanisms, purposes, and conclusions require source support.

Question:
{qa["question"]}

Answer:
{qa["answer"]}

Source:
---BEGIN SOURCE---
{source_excerpt(slot)}
---END SOURCE---
"""


def quality_prompt(slot, qa):
    return f"""Audit task routing and SFT quality for this candidate.

Assigned domain: {slot["domain"]}
Assigned task type: {slot["task_type"]}

Return ONLY:
{{
  "assigned_task_is_good_match": true,
  "best_task_type": "{slot["task_type"]}",
  "question_quality": 5,
  "answer_quality": 5,
  "quality_rationale": "<concise rationale>"
}}

Scoring:
- 5 = strong, clear, useful, nontrivial
- 4 = good with minor weakness
- 3 = usable but notable weakness
- 2 = poor
- 1 = unacceptable

Rules:
- assigned_task_is_good_match should be true only if the question genuinely exercises the assigned task.
- best_task_type must be a concise AxonAI task label.
- The answer must directly answer the question and remain precise.
- Prefer substantive scientific/technical reasoning over superficial wording.

Question:
{qa["question"]}

Answer:
{qa["answer"]}

Source excerpt:
---BEGIN SOURCE---
{source_excerpt(slot, 12000)}
---END SOURCE---
"""


def validate_grounding(obj):
    req(isinstance(obj.get("fully_grounded"), bool), "fully_grounded invalid")
    req(isinstance(obj.get("unsupported_claims"), list), "unsupported_claims invalid")


def validate_quality(obj):
    req(
        isinstance(obj.get("assigned_task_is_good_match"), bool),
        "assigned_task_is_good_match invalid",
    )
    req(isinstance(obj.get("best_task_type"), str), "best_task_type invalid")
    req(isinstance(obj.get("question_quality"), int), "question_quality invalid")
    req(isinstance(obj.get("answer_quality"), int), "answer_quality invalid")
    req(1 <= obj["question_quality"] <= 5, "question_quality out of range")
    req(1 <= obj["answer_quality"] <= 5, "answer_quality out of range")


def accepted(grounding, quality):
    return (
        grounding["fully_grounded"]
        and not grounding["unsupported_claims"]
        and quality["assigned_task_is_good_match"]
        and quality["question_quality"] >= 4
        and quality["answer_quality"] >= 4
    )


def rejection_feedback(grounding, quality):
    return {
        "fully_grounded": grounding["fully_grounded"],
        "unsupported_claims": grounding.get("unsupported_claims") or [],
        "grounding_rationale": grounding.get("grounding_rationale"),
        "assigned_task_is_good_match": quality["assigned_task_is_good_match"],
        "best_task_type": quality.get("best_task_type"),
        "question_quality": quality["question_quality"],
        "answer_quality": quality["answer_quality"],
        "quality_rationale": quality.get("quality_rationale"),
    }


def canonical_record(slot, qa, grounding, quality, attempt):
    record_id = (
        "axonai20k-"
        + hashlib.sha256(
            (slot["slot_id"] + "\0" + qa["question"]).encode("utf-8")
        ).hexdigest()[:24]
    )

    provenance = dict(slot.get("provenance") or {})
    provenance.update({
        "source_unit_id": slot["source_unit_id"],
        "source_document_id": slot["source_document_id"],
        "generator": GENERATOR_NAME,
        "verifier": VERIFIER_NAME,
        "generation_contract": VERSION,
        "split": "train",
        "source_family": slot["source_family"],
        "frozen_slot_id": slot["slot_id"],
        "generation_attempt": attempt,
    })

    for key in (
        "pmcid",
        "doi",
        "pmid",
        "primary_stratum",
        "source_chunk_ids",
        "repository",
        "commit",
        "source_file",
        "repository_snapshot",
    ):
        if slot.get(key) is not None:
            provenance[key] = slot.get(key)

    verification = {
        "fully_grounded": bool(grounding["fully_grounded"]),
        "unsupported_claims": grounding.get("unsupported_claims") or [],
        "answer_quality": int(quality["answer_quality"]),
        "best_task_type": quality["best_task_type"],
        "assigned_task_is_good_match": bool(
            quality["assigned_task_is_good_match"]
        ),
        "question_quality": int(quality["question_quality"]),
        "max_question_similarity": 0.0,
    }

    return {
        "id": record_id,
        "messages": [
            {"role": "system", "content": AXONAI_SYSTEM},
            {"role": "user", "content": qa["question"].strip()},
            {"role": "assistant", "content": qa["answer"].strip()},
        ],
        "domain": slot["domain"],
        "task_type": slot["task_type"],
        "source": slot.get("source"),
        "license": slot.get("license"),
        "quality_score": min(
            int(quality["question_quality"]),
            int(quality["answer_quality"]),
        ) / 5.0,
        "provenance": provenance,
        "verification": verification,
    }


def init_db(db_path: Path, manifest: list[dict], manifest_sha: str, max_attempts: int):
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")

    existing = dict(conn.execute("SELECT key, value FROM meta"))

    if existing:
        req(existing.get("version") == VERSION, "state version mismatch")
        req(existing.get("manifest_sha256") == manifest_sha, "manifest SHA mismatch")
        req(int(existing.get("max_attempts", "0")) == max_attempts, "max-attempts mismatch")
    else:
        conn.executemany(
            "INSERT INTO meta(key,value) VALUES (?,?)",
            [
                ("version", VERSION),
                ("manifest_sha256", manifest_sha),
                ("max_attempts", str(max_attempts)),
                ("created_at", str(now())),
            ],
        )

        ts = now()
        for ordinal, slot in enumerate(manifest, 1):
            conn.execute(
                """
                INSERT INTO slots(
                    slot_id, ordinal, source_json, attempts, status, updated_at
                ) VALUES (?,?,?,?,?,?)
                """,
                (
                    slot["slot_id"],
                    ordinal,
                    json.dumps(slot, ensure_ascii=False),
                    0,
                    "PENDING",
                    ts,
                ),
            )
        conn.commit()

    n = conn.execute("SELECT COUNT(*) FROM slots").fetchone()[0]
    req(n == len(manifest), f"state slots={n}, manifest={len(manifest)}")
    return conn


def status_counts(conn):
    return dict(conn.execute(
        "SELECT status, COUNT(*) FROM slots GROUP BY status ORDER BY status"
    ).fetchall())


def pending_generation_rows(conn, max_attempts):
    return conn.execute(
        """
        SELECT slot_id, ordinal, source_json, attempts, feedback_json, candidate_json
        FROM slots
        WHERE status IN ('PENDING','RETRY')
          AND attempts < ?
        ORDER BY ordinal
        """,
        (max_attempts,),
    ).fetchall()


def candidate_rows(conn):
    return conn.execute(
        """
        SELECT slot_id, ordinal, source_json, attempts, candidate_json
        FROM slots
        WHERE status='CANDIDATE'
        ORDER BY ordinal
        """
    ).fetchall()


def rebuild_outputs(conn, output_dir: Path):
    accepted_rows = conn.execute(
        """
        SELECT ordinal, canonical_json
        FROM slots
        WHERE status='ACCEPTED'
        ORDER BY ordinal
        """
    ).fetchall()

    accepted_path = output_dir / "accepted-canonical.jsonl"
    with accepted_path.open("w", encoding="utf-8") as f:
        for _, cjson in accepted_rows:
            f.write(cjson + "\n")

    exhausted_rows = conn.execute(
        """
        SELECT ordinal, slot_id, attempts, feedback_json, source_json
        FROM slots
        WHERE status='EXHAUSTED'
        ORDER BY ordinal
        """
    ).fetchall()

    exhausted_path = output_dir / "exhausted-slots.jsonl"
    with exhausted_path.open("w", encoding="utf-8") as f:
        for ordinal, slot_id, attempts, feedback_json, source_json in exhausted_rows:
            f.write(json.dumps({
                "ordinal": ordinal,
                "slot_id": slot_id,
                "attempts": attempts,
                "feedback": json.loads(feedback_json) if feedback_json else None,
                "source": json.loads(source_json),
            }, ensure_ascii=False) + "\n")

    return len(accepted_rows), len(exhausted_rows)


def print_progress(conn, stage, started, stage_done=None, stage_total=None):
    counts = status_counts(conn)
    accepted_n = counts.get("ACCEPTED", 0)
    total = sum(counts.values())
    elapsed = max(0.001, now() - started)
    rate = accepted_n / elapsed * 3600.0
    remaining = total - accepted_n
    eta_h = remaining / rate if rate > 0 else None

    extra = ""
    if stage_done is not None and stage_total is not None:
        extra = f" stage={stage_done}/{stage_total}"

    print(
        f"[PROGRESS] phase={stage}{extra} "
        f"accepted={accepted_n}/{total} "
        f"statuses={counts} "
        f"accepted_per_hour={rate:.2f} "
        f"ETA_hours={'?' if eta_h is None else f'{eta_h:.2f}'}",
        flush=True,
    )


def hard_final_audit(conn, manifest_sha):
    rows = conn.execute(
        """
        SELECT ordinal, slot_id, source_json, canonical_json
        FROM slots
        WHERE status='ACCEPTED'
        ORDER BY ordinal
        """
    ).fetchall()

    req(len(rows) == 7200, f"accepted={len(rows)}/7200")

    ids = set()
    questions = set()
    domains = Counter()
    tasks = Counter()
    families = Counter()

    for ordinal, slot_id, source_json, canonical_json in rows:
        slot = json.loads(source_json)
        rec = json.loads(canonical_json)

        req(set(rec) == REQUIRED_TOP_KEYS, f"{slot_id}: canonical schema keys")
        req(rec["provenance"]["frozen_slot_id"] == slot_id, f"{slot_id}: slot mismatch")
        req(rec["provenance"]["source_unit_id"] == slot["source_unit_id"], f"{slot_id}: source unit mismatch")
        req(rec["domain"] == slot["domain"], f"{slot_id}: domain mismatch")
        req(rec["task_type"] == slot["task_type"], f"{slot_id}: task mismatch")
        req(rec["verification"]["fully_grounded"], f"{slot_id}: not grounded")
        req(not rec["verification"]["unsupported_claims"], f"{slot_id}: unsupported claims")
        req(rec["verification"]["assigned_task_is_good_match"], f"{slot_id}: task mismatch verifier")
        req(rec["verification"]["question_quality"] >= 4, f"{slot_id}: q quality")
        req(rec["verification"]["answer_quality"] >= 4, f"{slot_id}: a quality")

        req(rec["id"] not in ids, f"{slot_id}: duplicate record id")
        ids.add(rec["id"])

        qnorm = re.sub(r"\s+", " ", rec["messages"][1]["content"]).strip().lower()
        req(qnorm not in questions, f"{slot_id}: exact duplicate question")
        questions.add(qnorm)

        domains[rec["domain"]] += 1
        tasks[(rec["domain"], rec["task_type"])] += 1
        families[rec["provenance"]["source_family"]] += 1

    return {
        "version": VERSION,
        "manifest_sha256": manifest_sha,
        "records": len(rows),
        "unique_record_ids": len(ids),
        "unique_exact_normalized_questions": len(questions),
        "domain_counts": dict(domains),
        "domain_task_counts": {
            f"{d}||{t}": n
            for (d, t), n in sorted(tasks.items())
        },
        "source_family_counts": dict(families),
        "fully_grounded_all": True,
        "unsupported_claims_none": True,
        "task_fit_all": True,
        "question_quality_min": 4,
        "answer_quality_min": 4,
        "status": "PRODUCTION_7200_ACCEPTED_HARD_AUDIT_PASS",
        "note": (
            "Near-duplicate similarity screening is intentionally deferred to the "
            "combined final corpus audit; this production audit checks exact question "
            "duplicates only."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--generator-model", type=Path, required=True)
    ap.add_argument("--verifier-model", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--progress-every", type=int, default=10)
    args = ap.parse_args()

    req(args.manifest.is_file(), f"missing manifest: {args.manifest}")
    req(args.generator_model.is_dir(), f"missing generator: {args.generator_model}")
    req(args.verifier_model.is_dir(), f"missing verifier: {args.verifier_model}")
    req(args.max_attempts >= 1, "max-attempts must be >=1")

    manifest = read_jsonl(args.manifest)
    req(len(manifest) == 7200, f"manifest records={len(manifest)}")
    req(len({r["slot_id"] for r in manifest}) == 7200, "duplicate manifest slot IDs")

    manifest_sha = sha256_file(args.manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    db_path = args.output_dir / "production-state.sqlite3"
    attempts_log = args.output_dir / "attempts-audit.jsonl"
    run_meta_path = args.output_dir / "run-metadata.json"

    conn = init_db(db_path, manifest, manifest_sha, args.max_attempts)

    run_meta = {
        "version": VERSION,
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha,
        "generator_model": str(args.generator_model),
        "verifier_model": str(args.verifier_model),
        "generator_name": GENERATOR_NAME,
        "verifier_name": VERIFIER_NAME,
        "max_attempts": args.max_attempts,
        "started_or_resumed_at": now(),
    }
    run_meta_path.write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    started = now()

    print("=" * 100)
    print("AXONAI +7,200 PRODUCTION GENERATION / VERIFICATION")
    print("=" * 100)
    print("Manifest:", args.manifest)
    print("Manifest SHA256:", manifest_sha)
    print("Output:", args.output_dir)
    print("Max attempts/slot:", args.max_attempts)
    print("Initial state:", status_counts(conn))
    print("Resume-safe: yes")
    print()

    cycle = 0

    while True:
        cycle += 1
        pending = pending_generation_rows(conn, args.max_attempts)

        if pending:
            print(f"\n=== CYCLE {cycle}: GENERATOR ({len(pending)} slots) ===", flush=True)
            gen_tok, gen_model = load_model(args.generator_model)

            stage_started = now()

            for idx, (slot_id, ordinal, source_json, attempts, feedback_json, previous_candidate_json) in enumerate(pending, 1):
                slot = json.loads(source_json)
                feedback = json.loads(feedback_json) if feedback_json else None
                previous_candidate = (
                    json.loads(previous_candidate_json)
                    if previous_candidate_json else None
                )
                attempt = attempts + 1

                try:
                    obj, raw = generate_json(
                        gen_tok,
                        gen_model,
                        [
                            {
                                "role": "system",
                                "content": (
                                    "You create rigorous source-grounded scientific "
                                    "SFT data. Return only valid JSON."
                                ),
                            },
                            {
                                "role": "user",
                                "content": generator_prompt(
                                    slot,
                                    attempt,
                                    feedback,
                                    previous_candidate,
                                ),
                            },
                        ],
                        max_new_tokens=1000,
                    )
                    validate_candidate(obj)

                    conn.execute(
                        """
                        UPDATE slots
                        SET attempts=?, status='CANDIDATE',
                            candidate_json=?, updated_at=?
                        WHERE slot_id=?
                        """,
                        (
                            attempt,
                            json.dumps(obj, ensure_ascii=False),
                            now(),
                            slot_id,
                        ),
                    )
                    conn.commit()

                    append_jsonl(attempts_log, {
                        "event": "generation",
                        "slot_id": slot_id,
                        "ordinal": ordinal,
                        "attempt": attempt,
                        "ok": True,
                        "candidate": obj,
                        "raw_output": raw,
                        "timestamp": now(),
                    })

                    print(
                        f"[GEN {ordinal:04d}/7200] {slot_id} "
                        f"attempt={attempt} OK",
                        flush=True,
                    )

                except KeyboardInterrupt:
                    raise

                except Exception as exc:
                    # Generation-format failures consume an attempt and retry later.
                    feedback_obj = {
                        "generation_error": repr(exc),
                        "instruction": "Return valid JSON with question and answer only.",
                    }

                    next_status = (
                        "EXHAUSTED"
                        if attempt >= args.max_attempts
                        else "RETRY"
                    )

                    conn.execute(
                        """
                        UPDATE slots
                        SET attempts=?, status=?, feedback_json=?, updated_at=?
                        WHERE slot_id=?
                        """,
                        (
                            attempt,
                            next_status,
                            json.dumps(feedback_obj, ensure_ascii=False),
                            now(),
                            slot_id,
                        ),
                    )
                    conn.commit()

                    append_jsonl(attempts_log, {
                        "event": "generation_error",
                        "slot_id": slot_id,
                        "ordinal": ordinal,
                        "attempt": attempt,
                        "ok": False,
                        "error": repr(exc),
                        "timestamp": now(),
                    })

                    print(
                        f"[GEN {ordinal:04d}/7200] {slot_id} "
                        f"attempt={attempt} ERROR {exc!r}",
                        flush=True,
                    )

                if idx % args.progress_every == 0:
                    print_progress(
                        conn,
                        "generation",
                        started,
                        idx,
                        len(pending),
                    )

            unload_model(gen_tok, gen_model)

        candidates = candidate_rows(conn)

        if candidates:
            print(f"\n=== CYCLE {cycle}: VERIFIER ({len(candidates)} candidates) ===", flush=True)
            ver_tok, ver_model = load_model(args.verifier_model)

            for idx, (slot_id, ordinal, source_json, attempts, candidate_json) in enumerate(candidates, 1):
                slot = json.loads(source_json)
                qa = json.loads(candidate_json)

                try:
                    grounding, grounding_raw = generate_json(
                        ver_tok,
                        ver_model,
                        [
                            {
                                "role": "system",
                                "content": (
                                    "You are a strict source-grounding verifier. "
                                    "Return only valid JSON."
                                ),
                            },
                            {
                                "role": "user",
                                "content": grounding_prompt(slot, qa),
                            },
                        ],
                        max_new_tokens=500,
                    )
                    validate_grounding(grounding)

                    quality, quality_raw = generate_json(
                        ver_tok,
                        ver_model,
                        [
                            {
                                "role": "system",
                                "content": (
                                    "You are a strict scientific SFT quality verifier. "
                                    "Return only valid JSON."
                                ),
                            },
                            {
                                "role": "user",
                                "content": quality_prompt(slot, qa),
                            },
                        ],
                        max_new_tokens=500,
                    )
                    validate_quality(quality)

                    ok = accepted(grounding, quality)

                    if ok:
                        canonical = canonical_record(
                            slot,
                            qa,
                            grounding,
                            quality,
                            attempts,
                        )

                        conn.execute(
                            """
                            UPDATE slots
                            SET status='ACCEPTED',
                                grounding_json=?,
                                quality_json=?,
                                feedback_json=NULL,
                                canonical_json=?,
                                updated_at=?
                            WHERE slot_id=?
                            """,
                            (
                                json.dumps(grounding, ensure_ascii=False),
                                json.dumps(quality, ensure_ascii=False),
                                json.dumps(canonical, ensure_ascii=False),
                                now(),
                                slot_id,
                            ),
                        )
                        conn.commit()

                        verdict = "ACCEPT"

                    else:
                        feedback = rejection_feedback(grounding, quality)
                        next_status = (
                            "EXHAUSTED"
                            if attempts >= args.max_attempts
                            else "RETRY"
                        )

                        conn.execute(
                            """
                            UPDATE slots
                            SET status=?,
                                grounding_json=?,
                                quality_json=?,
                                feedback_json=?,
                                updated_at=?
                            WHERE slot_id=?
                            """,
                            (
                                next_status,
                                json.dumps(grounding, ensure_ascii=False),
                                json.dumps(quality, ensure_ascii=False),
                                json.dumps(feedback, ensure_ascii=False),
                                now(),
                                slot_id,
                            ),
                        )
                        conn.commit()

                        verdict = next_status

                    append_jsonl(attempts_log, {
                        "event": "verification",
                        "slot_id": slot_id,
                        "ordinal": ordinal,
                        "attempt": attempts,
                        "accepted": ok,
                        "grounding": grounding,
                        "quality": quality,
                        "grounding_raw_output": grounding_raw,
                        "quality_raw_output": quality_raw,
                        "timestamp": now(),
                    })

                    print(
                        f"[VER {ordinal:04d}/7200] {slot_id} "
                        f"attempt={attempts} "
                        f"grounded={grounding['fully_grounded']} "
                        f"taskfit={quality['assigned_task_is_good_match']} "
                        f"q={quality['question_quality']} "
                        f"a={quality['answer_quality']} "
                        f"{verdict}",
                        flush=True,
                    )

                except KeyboardInterrupt:
                    raise

                except Exception as exc:
                    feedback = {
                        "verification_error": repr(exc),
                        "instruction": (
                            "Previous verifier execution failed; regenerate a concise "
                            "candidate that strictly follows the frozen task."
                        ),
                    }
                    next_status = (
                        "EXHAUSTED"
                        if attempts >= args.max_attempts
                        else "RETRY"
                    )

                    conn.execute(
                        """
                        UPDATE slots
                        SET status=?, feedback_json=?, updated_at=?
                        WHERE slot_id=?
                        """,
                        (
                            next_status,
                            json.dumps(feedback, ensure_ascii=False),
                            now(),
                            slot_id,
                        ),
                    )
                    conn.commit()

                    append_jsonl(attempts_log, {
                        "event": "verification_error",
                        "slot_id": slot_id,
                        "ordinal": ordinal,
                        "attempt": attempts,
                        "error": repr(exc),
                        "timestamp": now(),
                    })

                    print(
                        f"[VER {ordinal:04d}/7200] {slot_id} "
                        f"attempt={attempts} ERROR {exc!r}",
                        flush=True,
                    )

                if idx % args.progress_every == 0:
                    rebuild_outputs(conn, args.output_dir)
                    print_progress(
                        conn,
                        "verification",
                        started,
                        idx,
                        len(candidates),
                    )

            unload_model(ver_tok, ver_model)

        accepted_n, exhausted_n = rebuild_outputs(conn, args.output_dir)
        counts = status_counts(conn)

        print(
            f"\n=== END CYCLE {cycle} === "
            f"accepted={accepted_n} exhausted={exhausted_n} state={counts}",
            flush=True,
        )

        if accepted_n == 7200:
            break

        if exhausted_n > 0:
            print(
                "\nOne or more frozen slots exhausted the attempt budget. "
                "Stopping without source substitution.",
                flush=True,
            )
            break

        pending_n = counts.get("PENDING", 0) + counts.get("RETRY", 0)
        candidate_n = counts.get("CANDIDATE", 0)

        if pending_n == 0 and candidate_n == 0:
            raise RuntimeError(f"stalled state: {counts}")

    accepted_n, exhausted_n = rebuild_outputs(conn, args.output_dir)

    summary = {
        "version": VERSION,
        "manifest_sha256": manifest_sha,
        "accepted": accepted_n,
        "exhausted": exhausted_n,
        "status_counts": status_counts(conn),
        "attempt_distribution": dict(conn.execute(
            "SELECT attempts, COUNT(*) FROM slots GROUP BY attempts ORDER BY attempts"
        ).fetchall()),
        "elapsed_seconds": now() - started,
        "generator": GENERATOR_NAME,
        "verifier": VERIFIER_NAME,
    }

    if accepted_n == 7200 and exhausted_n == 0:
        audit = hard_final_audit(conn, manifest_sha)
        (args.output_dir / "production-hard-audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        summary["status"] = "PRODUCTION_7200_COMPLETE"
        summary["hard_audit_status"] = audit["status"]
    else:
        summary["status"] = "PRODUCTION_7200_INCOMPLETE"

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("PRODUCTION SUMMARY")
    print("=" * 100)
    print(json.dumps(summary, indent=2))

    conn.close()

    req(
        accepted_n == 7200 and exhausted_n == 0,
        (
            f"production incomplete: accepted={accepted_n}, "
            f"exhausted={exhausted_n}"
        ),
    )


if __name__ == "__main__":
    main()
