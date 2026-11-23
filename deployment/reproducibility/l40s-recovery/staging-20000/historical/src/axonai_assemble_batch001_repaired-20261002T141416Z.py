#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

VERSION = "batch001-repaired-5800-assembly-v1"
EXPECTED_RETAINED = 4870
EXPECTED_REFILLS = {"v1": 828, "v2": 82, "v3": 15, "v4": 4, "v5": 1}
EXPECTED_TOTAL = 5800
DOC_CAP = 12
UNIT_CAP = 2


def req(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_tar(path: Path):
    out = {}
    with tarfile.open(path, "r:gz") as tf:
        for member in tf:
            pp = PurePosixPath(member.name)
            req(
                not pp.is_absolute() and ".." not in pp.parts,
                f"unsafe archive member: {member.name}",
            )
            if member.isfile():
                f = tf.extractfile(member)
                req(f is not None, f"cannot read {member.name}")
                out[member.name] = f.read()
    return out


def jsonl_bytes(data: bytes):
    return [
        json.loads(line)
        for line in data.decode("utf-8").splitlines()
        if line.strip()
    ]


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_question(text: str) -> str:
    import re
    text = re.sub(r"\s+", " ", text.lower())
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return text.strip()


def dice_similarity(a: str, b: str) -> float:
    A = set(normalize_question(a).split())
    B = set(normalize_question(b).split())
    if not A or not B:
        return 0.0
    return 2.0 * len(A & B) / (len(A) + len(B))


def get_user_question(row):
    messages = row.get("messages")
    req(isinstance(messages, list), f"messages missing for {row.get('id')}")
    users = [
        m.get("content")
        for m in messages
        if isinstance(m, dict) and m.get("role") == "user"
    ]
    req(
        len(users) == 1 and isinstance(users[0], str),
        f"expected one user message for {row.get('id')}",
    )
    return users[0]


def get_system_message(row):
    messages = row.get("messages")
    req(isinstance(messages, list), "messages missing")
    systems = [
        m.get("content")
        for m in messages
        if isinstance(m, dict) and m.get("role") == "system"
    ]
    req(
        len(systems) == 1 and isinstance(systems[0], str),
        "expected one system message",
    )
    return systems[0]


def flatten_verification(refill):
    verification = refill.get("verification")
    req(isinstance(verification, dict), "refill verification missing")

    grounding = verification.get("grounding")
    task = verification.get("task_audit")

    req(isinstance(grounding, dict), "grounding verification missing")
    req(isinstance(task, dict), "task verification missing")

    unsupported = grounding.get("unsupported_claims")
    req(isinstance(unsupported, list), "unsupported_claims must be a list")

    result = {
        "fully_grounded": grounding.get("fully_grounded"),
        "unsupported_claims": unsupported,
        "answer_quality": grounding.get("answer_quality"),
        "best_task_type": task.get("best_task_type"),
        "assigned_task_is_good_match": task.get("assigned_task_is_good_match"),
        "question_quality": task.get("question_quality"),
        "max_question_similarity": (
            verification.get("max_question_similarity")
            if verification.get("max_question_similarity") is not None
            else refill.get("max_question_similarity_preverification", 0.0)
        ),
    }

    req(result["fully_grounded"] is True, "accepted refill is not fully grounded")
    req(result["unsupported_claims"] == [], "accepted refill has unsupported claims")
    req(isinstance(result["answer_quality"], int) and result["answer_quality"] >= 4,
        "accepted refill answer quality below threshold")
    req(result["assigned_task_is_good_match"] is True,
        "accepted refill assigned task mismatch")
    req(result["best_task_type"] == refill["task_type"],
        "accepted refill best task differs from assigned")
    req(isinstance(result["question_quality"], int) and result["question_quality"] >= 4,
        "accepted refill question quality below threshold")

    return result


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=Path, default=Path("/data/axonai"))
    ap.add_argument("--repair-plan", type=Path, required=True)

    ap.add_argument("--v1-review", type=Path, required=True)
    ap.add_argument("--v2-output-dir", type=Path, required=True)
    ap.add_argument("--v3-output-dir", type=Path, required=True)
    ap.add_argument("--v4-output-dir", type=Path, required=True)
    ap.add_argument("--v5-output-dir", type=Path, required=True)

    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--publish-archive", type=Path, required=True)

    args = ap.parse_args()

    root = args.root.resolve()

    original_path = (
        root
        / "datasets/axonai-v1/sft/train/batch001-5800.jsonl"
    )

    req(original_path.is_file(), f"missing original Batch 001: {original_path}")

    repair_tar = safe_tar(args.repair_plan)

    retained_rows = jsonl_bytes(
        repair_tar["plan/proposed-retained-ids.jsonl"]
    )

    retained_ids = {
        row.get("id") or row.get("example_id")
        for row in retained_rows
    }
    retained_ids.discard(None)

    req(
        len(retained_ids) == EXPECTED_RETAINED,
        f"expected {EXPECTED_RETAINED} retained IDs, got {len(retained_ids)}",
    )

    original = read_jsonl(original_path)

    req(
        len(original) == EXPECTED_TOTAL,
        f"original Batch 001 is not 5800 records: {len(original)}",
    )

    original_by_id = {
        row["id"]: row
        for row in original
    }

    req(
        len(original_by_id) == EXPECTED_TOTAL,
        "duplicate IDs exist in original Batch 001",
    )

    req(
        retained_ids <= set(original_by_id),
        "retained IDs missing from original Batch 001",
    )

    ###########################################################################
    # Canonical schema / constants derived from the original file itself
    ###########################################################################

    expected_top_keys = [
        "id",
        "messages",
        "domain",
        "task_type",
        "source",
        "license",
        "quality_score",
        "provenance",
        "verification",
    ]

    for row in original:
        req(
            list(row.keys()) == expected_top_keys,
            f"original top-level schema drift at {row.get('id')}",
        )

    system_messages = Counter(
        get_system_message(row)
        for row in original
    )

    req(
        len(system_messages) == 1,
        f"original dataset has multiple system prompts: {len(system_messages)}",
    )

    canonical_system = next(iter(system_messages))

    quality_scores = Counter(
        row.get("quality_score")
        for row in original
    )

    req(
        len(quality_scores) == 1,
        f"original dataset has multiple quality_score conventions: {quality_scores}",
    )

    canonical_quality_score = next(iter(quality_scores))

    provenance_constants = {}

    for field in (
        "generator",
        "verifier",
        "generation_contract",
        "split",
    ):
        values = Counter(
            row["provenance"].get(field)
            for row in original
        )
        req(
            len(values) == 1,
            f"original provenance.{field} is not constant: {values}",
        )
        provenance_constants[field] = next(iter(values))

    req(
        provenance_constants["split"] == "train",
        "original Batch 001 split is not train",
    )

    ###########################################################################
    # Load exact accepted refill sets
    ###########################################################################

    v1_review = safe_tar(args.v1_review)
    v1 = jsonl_bytes(v1_review["./run/accepted.jsonl"])

    refill_paths = {
        "v2": args.v2_output_dir / "accepted.jsonl",
        "v3": args.v3_output_dir / "accepted.jsonl",
        "v4": args.v4_output_dir / "accepted.jsonl",
        "v5": args.v5_output_dir / "accepted.jsonl",
    }

    refills = {
        "v1": v1,
        **{
            name: read_jsonl(path)
            for name, path in refill_paths.items()
        },
    }

    for name, expected in EXPECTED_REFILLS.items():
        req(
            len(refills[name]) == expected,
            f"{name} accepted count mismatch: {len(refills[name])} != {expected}",
        )

    req(
        sum(len(v) for v in refills.values()) == 930,
        "total refill accepted count !=930",
    )

    ###########################################################################
    # Retained records: preserve exactly, in original order
    ###########################################################################

    retained_output = [
        row
        for row in original
        if row["id"] in retained_ids
    ]

    req(
        len(retained_output) == EXPECTED_RETAINED,
        "retained output count mismatch",
    )

    ###########################################################################
    # Convert accepted refill audit records to canonical SFT schema
    ###########################################################################

    converted = []

    refill_id_seen = set()

    for version in ("v1", "v2", "v3", "v4", "v5"):
        for refill in refills[version]:

            allocation_id = refill.get("allocation_id")

            req(
                isinstance(allocation_id, str) and allocation_id,
                f"{version} refill missing allocation_id",
            )

            canonical_id = (
                "axonai-batch001-repair-"
                + allocation_id
            )

            req(
                canonical_id not in original_by_id,
                f"replacement ID collides with original: {canonical_id}",
            )

            req(
                canonical_id not in refill_id_seen,
                f"duplicate replacement ID: {canonical_id}",
            )

            refill_id_seen.add(canonical_id)

            question = refill.get("question")
            answer = refill.get("answer")

            req(
                isinstance(question, str) and question.strip(),
                f"{canonical_id}: question missing",
            )

            req(
                isinstance(answer, str) and answer.strip(),
                f"{canonical_id}: answer missing",
            )

            source_id = refill.get("source_id")
            source_unit_id = refill.get("source_unit_id")
            source_file = refill.get("source_file")

            req(
                all(
                    isinstance(x, str) and x
                    for x in (
                        source_id,
                        source_unit_id,
                        source_file,
                    )
                ),
                f"{canonical_id}: provenance identity incomplete",
            )

            licence = refill.get("applicable_license")

            req(
                isinstance(licence, str) and licence,
                f"{canonical_id}: applicable licence missing",
            )

            verification = flatten_verification(refill)

            provenance = {
                "source_id":
                    source_id,

                "source_unit_id":
                    source_unit_id,

                "source_document_id":
                    source_file,

                "generator":
                    provenance_constants["generator"],

                "verifier":
                    provenance_constants["verifier"],

                "generation_contract":
                    provenance_constants["generation_contract"],

                "split":
                    "train",
            }

            canonical = {
                "id":
                    canonical_id,

                "messages": [
                    {
                        "role": "system",
                        "content": canonical_system,
                    },
                    {
                        "role": "user",
                        "content": question.strip(),
                    },
                    {
                        "role": "assistant",
                        "content": answer.strip(),
                    },
                ],

                "domain":
                    refill["domain"],

                "task_type":
                    refill["task_type"],

                "source":
                    source_id,

                "license":
                    licence,

                "quality_score":
                    canonical_quality_score,

                "provenance":
                    provenance,

                "verification":
                    verification,
            }

            req(
                list(canonical.keys()) == expected_top_keys,
                "canonical replacement schema mismatch",
            )

            converted.append(canonical)

    req(
        len(converted) == 930,
        f"converted refill count !=930: {len(converted)}",
    )

    ###########################################################################
    # Assemble 5,800
    ###########################################################################

    final_rows = retained_output + converted

    req(
        len(final_rows) == EXPECTED_TOTAL,
        f"assembled count !=5800: {len(final_rows)}",
    )

    ids = [row["id"] for row in final_rows]

    req(
        len(ids) == len(set(ids)),
        "assembled dataset contains duplicate IDs",
    )

    ###########################################################################
    # HARD AUDITS
    ###########################################################################

    # Schema / roles / train split
    for row in final_rows:

        req(
            list(row.keys()) == expected_top_keys,
            f"schema mismatch: {row['id']}",
        )

        roles = [
            m.get("role")
            for m in row["messages"]
        ]

        req(
            roles == ["system", "user", "assistant"],
            f"message roles mismatch: {row['id']}",
        )

        req(
            row["provenance"].get("split") == "train",
            f"non-train record: {row['id']}",
        )

        req(
            row["verification"].get("fully_grounded") is True,
            f"not fully grounded: {row['id']}",
        )

        req(
            row["verification"].get("unsupported_claims") == [],
            f"unsupported claims: {row['id']}",
        )

    # Exact question duplicates
    questions = [
        get_user_question(row)
        for row in final_rows
    ]

    normalized = [
        normalize_question(q)
        for q in questions
    ]

    normalized_counts = Counter(normalized)

    req(
        max(normalized_counts.values(), default=0) == 1,
        "exact normalized duplicate question detected",
    )

    # Near-duplicate audit.
    # The refill pipeline already screened replacement candidates, but the final
    # assembly rechecks the full file conservatively and records the maximum.
    max_similarity = 0.0
    max_pair = None

    # Use inverted token index to avoid full 5800^2 comparison where possible.
    token_sets = [
        set(x.split())
        for x in normalized
    ]

    inverted = defaultdict(list)

    for i, tokens in enumerate(token_sets):
        candidate_indices = set()

        for token in tokens:
            candidate_indices.update(
                inverted[token]
            )

        for j in candidate_indices:
            similarity = (
                0.0
                if not tokens or not token_sets[j]
                else (
                    2.0
                    * len(tokens & token_sets[j])
                    / (
                        len(tokens)
                        + len(token_sets[j])
                    )
                )
            )

            if similarity > max_similarity:
                max_similarity = similarity
                max_pair = (
                    final_rows[j]["id"],
                    final_rows[i]["id"],
                )

            req(
                similarity < 0.90,
                (
                    "near-duplicate >=0.90 detected: "
                    f"{final_rows[j]['id']} vs "
                    f"{final_rows[i]['id']} "
                    f"({similarity:.4f})"
                ),
            )

        for token in tokens:
            inverted[token].append(i)

    # True-document / source-unit caps
    doc_counts = Counter()
    unit_counts = Counter()

    for row in final_rows:
        p = row["provenance"]

        source_id = p["source_id"]
        source_document_id = p["source_document_id"]
        source_unit_id = p["source_unit_id"]

        doc_key = (
            source_id,
            source_document_id,
        )

        unit_key = (
            source_id,
            source_document_id,
            source_unit_id,
        )

        doc_counts[doc_key] += 1
        unit_counts[unit_key] += 1

    req(
        max(doc_counts.values(), default=0) <= DOC_CAP,
        f"document cap exceeded: {max(doc_counts.values())}",
    )

    req(
        max(unit_counts.values(), default=0) <= UNIT_CAP,
        f"source-unit cap exceeded: {max(unit_counts.values())}",
    )

    # Task/domain counts
    task_counts = Counter(
        (
            row["domain"],
            row["task_type"],
        )
        for row in final_rows
    )

    domain_counts = Counter(
        row["domain"]
        for row in final_rows
    )

    ###########################################################################
    # Write versioned artifact; original remains untouched
    ###########################################################################

    out = args.output_dir

    req(
        not out.exists(),
        f"output directory already exists: {out}",
    )

    out.mkdir(parents=True)

    assembled_path = (
        out
        / "batch001-5800-repaired.jsonl"
    )

    with assembled_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in final_rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    report = {
        "version":
            VERSION,

        "original_batch001":
            str(original_path),

        "original_batch001_sha256":
            sha256_file(original_path),

        "repaired_batch001":
            str(assembled_path),

        "records":
            len(final_rows),

        "composition": {
            "retained_original":
                len(retained_output),

            "replacement_v1":
                len(refills["v1"]),

            "replacement_v2":
                len(refills["v2"]),

            "replacement_v3":
                len(refills["v3"]),

            "replacement_v4":
                len(refills["v4"]),

            "replacement_v5":
                len(refills["v5"]),
        },

        "unique_ids":
            len(set(ids)),

        "exact_normalized_question_duplicates":
            0,

        "near_duplicate_threshold":
            0.90,

        "maximum_observed_question_similarity_below_threshold":
            max_similarity,

        "maximum_similarity_pair":
            max_pair,

        "maximum_examples_per_document":
            max(
                doc_counts.values(),
                default=0,
            ),

        "maximum_examples_per_source_unit":
            max(
                unit_counts.values(),
                default=0,
            ),

        "domain_counts":
            dict(
                sorted(
                    domain_counts.items()
                )
            ),

        "task_counts": {
            f"{domain}/{task}":
                count
            for (
                domain,
                task,
            ), count
            in sorted(
                task_counts.items()
            )
        },

        "canonical_system_message":
            canonical_system,

        "canonical_quality_score":
            canonical_quality_score,

        "provenance_constants":
            provenance_constants,

        "status":
            "REPAIRED_BATCH001_HARD_AUDIT_PASS",
    }

    report_path = (
        out
        / "hard-audit-report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "version":
            VERSION,

        "inputs": {
            "original_batch001_sha256":
                sha256_file(
                    original_path
                ),

            "repair_plan_sha256":
                sha256_file(
                    args.repair_plan
                ),

            "v1_review_sha256":
                sha256_file(
                    args.v1_review
                ),

            "v2_accepted_sha256":
                sha256_file(
                    refill_paths["v2"]
                ),

            "v3_accepted_sha256":
                sha256_file(
                    refill_paths["v3"]
                ),

            "v4_accepted_sha256":
                sha256_file(
                    refill_paths["v4"]
                ),

            "v5_accepted_sha256":
                sha256_file(
                    refill_paths["v5"]
                ),
        },

        "output_sha256":
            sha256_file(
                assembled_path
            ),
    }

    manifest_path = (
        out
        / "input-output-manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with (
        out
        / "SHA256SUMS"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        for p in (
            assembled_path,
            report_path,
            manifest_path,
        ):
            f.write(
                f"{sha256_file(p)}  "
                f"{p.name}\n"
            )

    req(
        not args.publish_archive.exists(),
        f"publish archive already exists: {args.publish_archive}",
    )

    with tarfile.open(
        args.publish_archive,
        "w:gz",
    ) as tf:
        for p in sorted(
            out.iterdir()
        ):
            if p.is_file():
                tf.add(
                    p,
                    arcname=(
                        "batch001-repaired/"
                        + p.name
                    ),
                )

    print("=" * 88)
    print("AXONAI BATCH 001 — REPAIRED CANONICAL ASSEMBLY")
    print("=" * 88)
    print("Retained original:", len(retained_output))
    print("V1 replacements:", len(refills["v1"]))
    print("V2 replacements:", len(refills["v2"]))
    print("V3 replacements:", len(refills["v3"]))
    print("V4 replacements:", len(refills["v4"]))
    print("V5 replacements:", len(refills["v5"]))
    print("Final records:", len(final_rows))
    print("Unique IDs:", len(set(ids)))
    print("Exact normalized duplicate questions: 0")
    print(
        "Maximum near-question similarity:",
        f"{max_similarity:.4f}",
    )
    print(
        "Maximum examples/document:",
        max(doc_counts.values(), default=0),
    )
    print(
        "Maximum examples/source-unit:",
        max(unit_counts.values(), default=0),
    )
    print("All records train split: YES")
    print("All records fully grounded: YES")
    print("Unsupported claims present: NO")
    print("Original Batch 001 overwritten: NO")
    print()
    print("Status: REPAIRED_BATCH001_HARD_AUDIT_PASS")


if __name__ == "__main__":
    main()
