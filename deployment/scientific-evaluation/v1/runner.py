"""Prospective pilot worker protocol. Pure Python; no launch or model-loading entry point."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re


PROVENANCE_SHA256 = "f159c32a878ed623dc7cb98cb48f50d25f1fc5d16368811b8ebe00edcf533e9f"
BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
CONTEXT = 4096
NEW_TOKENS = 384
SYSTEM_MESSAGE = "You are a scientific research assistant. Follow the requested answer format."
FAILURES = frozenset({"context_overflow", "oom", "timeout", "nonfinite_logits",
                      "backend_error", "worker_lost", "empty_output"})


class IntegrityError(ValueError):
    """Halt the run and investigate before scoring."""


def require(ok, problem):
    if not ok:
        raise IntegrityError(problem)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha(encoded.encode("utf-8"))


def generation_profile(eos_ids):
    """Values the approved worker must set explicitly and bind in its kernel configuration."""
    return {"dtype": "bfloat16", "attention": "sdpa", "sdpa_backend": "math",
            "placement": "frozen_two_gpu_map", "batch_size": 1, "seed": 42,
            "context_limit": CONTEXT, "max_new_tokens": NEW_TOKENS, "do_sample": False,
            "num_beams": 1, "repetition_penalty": 1.0, "use_cache": True,
            "stop_strings": [], "eos_token_ids": list(eos_ids), "allow_tf32": False,
            "deterministic_algorithms": True, "cudnn_benchmark": False,
            "enable_thinking": False, "add_generation_prompt": True, "add_special_tokens": False}


def valid_digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def opaque(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{32}", value) is not None


def verify_directory(directory, entries):
    """Custodian only. Verify a dedicated read-only mount with exactly the selected files."""
    names = {Path(e["path"]).name for e in entries}
    require(len(names) == len(entries) and names == {p.name for p in directory.iterdir()},
            "artifact inventory mismatch")
    for entry in entries:
        path = directory / Path(entry["path"]).name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == entry["bytes"],
                "artifact type or size mismatch")
        with path.open("rb") as stream:
            hasher = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
            require(hasher.hexdigest() == entry["sha256"], "artifact bytes mismatch")


def verify_identity(provenance, base_mount, adapter_mount=None):
    """Custodian preflight for one primary arm. Keep the returned identity receipt private."""
    raw = provenance.read_bytes()
    require(sha(raw) == PROVENANCE_SHA256, "wrong provenance manifest")
    record = json.loads(raw)
    require(record["base_revision"] == BASE_REVISION, "wrong base revision")
    base_files = [e for e in record["base_files"]
                  if Path(e["path"]).parent == Path("/data/axonai/models/base/Qwen3.8-27B")]
    require(sum(e["path"].endswith(".safetensors") for e in base_files) == 18, "shard inventory")
    verify_directory(base_mount, base_files)
    if adapter_mount is not None:
        verify_directory(adapter_mount, [record["adapter"], record["adapter_config"]])
    return {"base_revision": BASE_REVISION, "provenance_sha256": PROVENANCE_SHA256,
            "base_inventory_sha256": digest(base_files),
            "adapter_sha256": record["adapter"]["sha256"] if adapter_mount is not None else None}


@dataclass(frozen=True)
class Runtime:
    """Common, *observed* runtime bindings published privately by each worker."""
    image_sha256: str
    software_sha256: str
    device_map_sha256: str
    kernel_config_sha256: str
    renderer_sha256: str
    eos_ids: tuple
    ordered_l40s_uuids: tuple

    def validate(self):
        for name in ("image_sha256", "software_sha256", "device_map_sha256",
                     "kernel_config_sha256", "renderer_sha256"):
            require(valid_digest(getattr(self, name)), "missing common runtime evidence")
        require(type(self.eos_ids) is tuple and self.eos_ids
                and len(set(self.eos_ids)) == len(self.eos_ids)
                and all(type(i) is int and i >= 0 for i in self.eos_ids), "invalid EOS inventory")
        pattern = r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
        require(type(self.ordered_l40s_uuids) is tuple and len(self.ordered_l40s_uuids) == 2
                and len(set(self.ordered_l40s_uuids)) == 2
                and all(re.fullmatch(pattern, gpu) for gpu in self.ordered_l40s_uuids),
                "expected two distinct ordered device UUIDs")


@dataclass(frozen=True)
class Item:
    item_id: str
    rendered_sha256: str
    input_ids: tuple

    def validate(self):
        require(opaque(self.item_id) and valid_digest(self.rendered_sha256), "invalid item binding")
        require(type(self.input_ids) is tuple and self.input_ids
                and all(type(i) is int and i >= 0 for i in self.input_ids), "invalid input IDs")


def render_item(item_id, user_text, tokenizer):
    """Render once in the pilot-only renderer. Persist the exact IDs for both slots."""
    require(isinstance(user_text, str) and user_text.strip(), "empty private question")
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": user_text}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    ids = tokenizer(rendered, add_special_tokens=False, truncation=False, padding=False)["input_ids"]
    item = Item(item_id, sha(rendered.encode("utf-8")), tuple(ids))
    item.validate()
    return item


@dataclass(frozen=True)
class Measurement:
    latency_s: float
    first_token_s: float | None
    peak_allocated_bytes: tuple
    peak_reserved_bytes: tuple
    peak_cpu_rss_bytes: int

    def validate(self):
        require(type(self.latency_s) in (float, int) and math.isfinite(self.latency_s)
                and self.latency_s > 0, "invalid latency")
        first = self.first_token_s
        require(first is None or type(first) in (int, float) and math.isfinite(first)
                and 0 <= first <= self.latency_s, "invalid first-token latency")
        for counts in (self.peak_allocated_bytes, self.peak_reserved_bytes):
            require(type(counts) is tuple and len(counts) == 2
                    and all(type(x) is int and x >= 0 for x in counts), "invalid per-GPU peaks")
        require(all(a <= b for a, b in zip(self.peak_allocated_bytes, self.peak_reserved_bytes)),
                "allocated peak exceeds reserved peak")
        require(type(self.peak_cpu_rss_bytes) is int and self.peak_cpu_rss_bytes > 0,
                "invalid host peak")


@dataclass(frozen=True)
class Decode:
    """Backend computes input_sha256 from the actual device-bound tensor and outputs at batch size 1."""
    input_sha256: str
    output_ids: tuple
    text: str
    measurement: Measurement


class ItemFailure(Exception):
    """Typed backend failure. Do not forward exception strings or traces to the orchestrator."""
    def __init__(self, status, measurement=None):
        if status not in {"oom", "timeout", "nonfinite_logits", "backend_error", "worker_lost"}:
            raise ValueError("unknown worker failure class")
        super().__init__(status)
        self.status = status
        self.measurement = measurement


def run_item(run_id, slot_ticket, item, expected_runtime, observed_runtime, decode):
    """Run one scheduled private job. Return its immutable-record payload to the private output sink.

    expected_runtime comes from the approved freeze, observed_runtime from an authenticated worker.
    decode accepts the exact input ID tuple and the fixed new-token limit. It must attest its
    actual input at the model-call boundary. Identity/parity errors propagate and stop the run.
    """
    require(opaque(run_id) and opaque(slot_ticket), "invalid run or ticket")
    expected_runtime.validate()
    require(expected_runtime == observed_runtime, "worker runtime differs from freeze")
    item.validate()
    token_digest = digest(item.input_ids)
    row = {"schema_version": 1, "run_id": run_id, "slot_ticket": slot_ticket, "item_id": item.item_id,
           "runtime_sha256": digest(asdict(observed_runtime)),
           "generation_sha256": digest(generation_profile(observed_runtime.eos_ids)),
           "runner_sha256": sha(Path(__file__).read_bytes()), "rendered_sha256": item.rendered_sha256,
           "input_sha256": token_digest, "input_tokens": len(item.input_ids),
           "context_limit": CONTEXT, "new_token_limit": NEW_TOKENS,
           "actual_input_sha256": None, "output_tokens": None, "output_ids": None, "text": None,
           "output_sha256": None, "stop_reason": None, "measurement": None}

    if len(item.input_ids) + NEW_TOKENS > CONTEXT:
        return {**row, "status": "context_overflow"}
    try:
        result = decode(item.input_ids, NEW_TOKENS)
    except ItemFailure as error:
        if error.measurement is not None:
            error.measurement.validate()
            row["measurement"] = asdict(error.measurement)
        return {**row, "status": error.status}

    require(result.input_sha256 == token_digest, "actual model input changed")
    result.measurement.validate()
    ids = result.output_ids
    require(type(ids) is tuple and 0 < len(ids) <= NEW_TOKENS
            and all(type(x) is int and x >= 0 for x in ids), "invalid generated IDs")
    eos = expected_runtime.eos_ids
    require(not any(t in eos for t in ids[:-1]), "generation continued after EOS")
    require(ids[-1] in eos or len(ids) == NEW_TOKENS, "unexplained early stop")
    require(result.measurement.first_token_s is not None and isinstance(result.text, str),
            "incomplete generation measurement")
    stopped_at_eos = ids[-1] in eos
    status = "ok" if stopped_at_eos else "length_limit"
    if not result.text.strip():
        status = "empty_output"
    return {**row, "actual_input_sha256": result.input_sha256, "status": status,
            "stop_reason": "eos" if stopped_at_eos else "length_limit",
            "output_tokens": len(ids), "output_ids": ids, "text": result.text,
            "output_sha256": digest([ids, result.text]), "measurement": asdict(result.measurement)}


def audit_pairs(schedule, receipts, run_id, tickets, frozen_runtime):
    """Private custodian reconciles the complete predeclared roster before releasing blinded scores."""
    require(opaque(run_id) and len(tickets) == 2 and len(set(tickets)) == 2
            and all(opaque(t) for t in tickets), "invalid ticket allocation")
    frozen_runtime.validate()
    runtime_digest = digest(asdict(frozen_runtime))
    generation_digest = digest(generation_profile(frozen_runtime.eos_ids))
    runner_digest = sha(Path(__file__).read_bytes())
    planned = {}
    for item in schedule:
        item.validate()
        require(item.item_id not in planned, "duplicate scheduled item")
        planned[item.item_id] = (item.rendered_sha256, digest(item.input_ids), len(item.input_ids))
    require(planned, "empty roster")
    expected = {(item_id, ticket) for item_id in planned for ticket in tickets}
    seen = set()
    for row in receipts:
        key = (row["item_id"], row["slot_ticket"])
        require(key in expected and key not in seen, "duplicate or unscheduled receipt")
        seen.add(key)
        require(row["schema_version"] == 1 and row["run_id"] == run_id
                and row["runtime_sha256"] == runtime_digest and row["runner_sha256"] == runner_digest
                and row["generation_sha256"] == generation_digest
                and (row["context_limit"], row["new_token_limit"]) == (CONTEXT, NEW_TOKENS),
                "receipt does not match run freeze")
        require(type(row["input_tokens"]) is int
                and (row["rendered_sha256"], row["input_sha256"], row["input_tokens"]) == planned[key[0]],
                "receipt does not match rendered item")
        over = row["input_tokens"] + NEW_TOKENS > CONTEXT
        status = row["status"]
        require((status == "context_overflow") == over, "context handling mismatch")
        if status in {"ok", "length_limit", "empty_output"}:
            ids = row["output_ids"]
            require(row["actual_input_sha256"] == row["input_sha256"]
                    and type(row["output_tokens"]) is int and isinstance(ids, (list, tuple))
                    and 0 < len(ids) == row["output_tokens"] <= NEW_TOKENS
                    and all(type(x) is int and x >= 0 for x in ids)
                    and isinstance(row["text"], str) and digest([ids, row["text"]]) == row["output_sha256"],
                    "incomplete or inconsistent response")
            eos = frozen_runtime.eos_ids
            require(not any(t in eos for t in ids[:-1]) and (ids[-1] in eos or len(ids) == NEW_TOKENS),
                    "invalid stop condition")
            expected_status = "ok" if ids[-1] in eos else "length_limit"
            require(row["stop_reason"] == ("eos" if ids[-1] in eos else "length_limit")
                    and status == (expected_status if row["text"].strip() else "empty_output")
                    and row["measurement"] is not None, "response outcome mismatch")
        else:
            require(status in FAILURES and all(row[k] is None for k in
                    ("actual_input_sha256", "output_sha256", "output_tokens", "output_ids", "text",
                     "stop_reason")),
                    "invalid failure receipt")
        if row["measurement"] is not None:
            metrics = dict(row["measurement"])
            for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
                metrics[name] = tuple(metrics[name])
            measurement = Measurement(**metrics)
            measurement.validate()
            require(row["output_ids"] is None or measurement.first_token_s is not None,
                    "missing first-token measurement")
    require(seen == expected, "missing receipt; reconcile every scheduled slot")
    return len(seen)
