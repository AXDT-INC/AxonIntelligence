"""Offline checks of AxonAI-v1 release metadata and optional original inputs."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


MODELS = Path(__file__).resolve().parents[2] / "models" / "merged"
BF16 = MODELS / "AxonAI-v1-27B"
FP16 = MODELS / "AxonAI-v1-27B-FP16"
MANIFEST = "SHA256_MANIFEST.json"
ATTRIBUTIONS = "SOURCE_ATTRIBUTIONS.inherited-draft.json"
BF16_DIGEST = "d34c6389c0ca68df59e11f2a18f3616e7b335066fd8c493acf46044551e1daee"
FP16_DIGEST = "c3a6cb1deb5123cc35ccc9c07de36545f82b91a317f97971b6bd56b3403f2e06"
SNAPSHOT = Path("engineering-maintenance/fsdp1-main-execution-snapshot-v4-20261029")
SNAPSHOT_DIGEST = "5cce07110d230da8fcb84b8823459d5b7d89d3d597398395b2acbc5addb4a93d"


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify_file(path, digest, size=None):
    if size is not None:
        check(path.stat().st_size == size, f"Size mismatch: {path}")
    actual = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            actual.update(chunk)
    check(actual.hexdigest() == digest, f"SHA-256 mismatch: {path}")


def manifest(root, digest):
    path = root / MANIFEST
    verify_file(path, digest)
    rows = json.loads(path.read_bytes())["files"]
    return {row["relative_path"]: row for row in rows}


def read_record(root, rows, name):
    row = rows[name]
    verify_file(root / name, row["sha256"], row["bytes"])
    return json.loads((root / name).read_bytes())


def release_records():
    bf = manifest(BF16, BF16_DIGEST)
    fp = manifest(FP16, FP16_DIGEST)
    provenance = read_record(BF16, bf, "PROVENANCE.json")
    attribution = read_record(BF16, bf, ATTRIBUTIONS)
    sources = attribution["sources"]
    source_ids = {r["pmcid"] for r in sources}
    check(len(source_ids) == len(sources) == attribution["source_count"] == 4230,
          "Source inventory differs")
    check(sum(r["training_record_count"] for r in sources) == 10684,
          "Attributed record count differs")
    check(Counter(r["license"] for r in sources) == {"CC-BY": 4175, "CC0": 55},
          "Source license counts differ")

    merge = read_record(BF16, bf, "MERGE_EQUATION_VERIFICATION.json")
    validation = read_record(BF16, bf, "VALIDATION_SUMMARY.json")
    conversion = read_record(FP16, fp, ".conversion/verification.json")
    check(merge == validation["all_weight_equations_verified"], "Merge reports disagree")
    check(merge["updated_tensors_exactly_match_fp32_merge_then_bf16"] == 496
          and merge["unadapted_tensors_match_loaded_base"] == 355
          and merge["all_adapter_tensors_consumed"] == 992, "Merge inventory differs")
    base_map = read_record(BF16, bf, "model.safetensors.index.json")["weight_map"]
    output_map = read_record(FP16, fp, "model.safetensors.index.json")["weight_map"]
    check(base_map == output_map and len(base_map) == conversion["converted"] == 851,
          "Weight indexes or conversion counts differ")
    shards = set(base_map.values())
    check(shards == {p for p in bf if p.endswith(".safetensors")}
          == {p for p in fp if p.endswith(".safetensors")}
          and len(shards) == conversion["shards"] == 14, "Shard inventory differs")
    check(conversion["source_manifest_sha256"] == BF16_DIGEST
          and conversion["elements"] == provenance["merge"]["parameter_count"] == 26895998464
          and conversion["status"] == "passed", "Conversion report differs")
    print("Release metadata verified: 4,230 sources, 10,684 attributed records,"
          " 496 adapted weights, 851 exported tensors and 14 shards per release.")
    return provenance, attribution


def archive_inputs(root, groups, provenance, attribution):
    # The pinned historical list named base_files also contains both adapter files.
    rows = {
        "dataset": [provenance["dataset"]],
        "attribution": [attribution["source"]],
        "config": [provenance["adapter_config"]],
        "adapter": [provenance["adapter"]],
        "base": [r for r in provenance["base_files"]
                 if Path(r["path"]).is_relative_to("/data/axonai/models/base")],
    }
    for group in dict.fromkeys(groups):
        for row in rows[group]:
            relative = Path(row["path"]).relative_to("/data/axonai")
            verify_file(root / relative, row["sha256"], row["bytes"])
        print(f"Original {group} verified: {len(rows[group])} file(s).")


def recovered_sources(root, provenance):
    config = provenance["adapter_config"]
    path = Path(config["path"]).relative_to("/data/axonai")
    verify_file(root / path, config["sha256"], config["bytes"])
    snapshot = root / SNAPSHOT / "SNAPSHOT.json"
    verify_file(snapshot, SNAPSHOT_DIGEST)
    members = json.loads(snapshot.read_bytes())["files"]
    for name, digest in members.items():
        verify_file(snapshot.parent / name, digest)
    run = (root / path).parent.parent
    identity = json.loads((run / "identity.json").read_bytes())
    auth = json.loads((run / "LAUNCH_AUTHORIZATION.json").read_bytes())
    result = json.loads((run / "TRAINING_RESULT.json").read_bytes())
    bindings = result["bindings"]
    verify_file(run / "identity.json", auth["identity_sha256"])
    approval_path = root / "engineering-maintenance/project-owner-governance-v1-20261029/APPROVED_MAIN_SFT.v3.json"
    check(Path(auth["approval"]) == Path("/data/axonai") / approval_path.relative_to(root),
          "Launch authorization names a different approval")
    verify_file(approval_path, auth["approval_sha256"])
    approval = json.loads(approval_path.read_bytes())
    check(auth["approval_sha256"] == identity["approval_sha256"]
          and auth["identity_sha256"] == bindings["identity_sha256"], "Launch identity differs")
    check(approval["snapshot_manifest"]["path"] == str(Path("/data/axonai") / SNAPSHOT / snapshot.name)
          and approval["snapshot_manifest"]["sha256"] == identity["snapshot_sha256"] == SNAPSHOT_DIGEST
          and identity["code"] == members, "Execution snapshot differs")
    check(approval["worker_sha256"] == bindings["worker_sha256"] == members["train.py"]
          and approval["config_sha256"] == identity["config_sha256"] == bindings["config_sha256"]
          == members["config.fsdp1.json"], "Executed worker or configuration differs")
    events = [json.loads(line) for line in (run / "lifecycle.jsonl").read_bytes().splitlines()]
    start, = [e for e in events if e["event"] == "train_started"]
    finish, = [e for e in events if e["event"] == "train_finished"]
    check(start["bindings"] == bindings and start["global_step"] == 0 and start["resume_from"] is None
          and start["validation_examples"] == 0 and start["train_examples"] == 10684,
          "Training start differs")
    check(finish == dict(result, event="train_finished") and result["success"] is True
          and result["stop_requested"] is False and result["resumed_from"] is None
          and result["global_step"] == approval["optimizer_steps"] == 668, "Training completion differs")
    contract = json.loads((snapshot.parent / "contract.json").read_bytes())["bindings"]
    receipt = contract["extraction"]
    check(auth["extraction_receipt"] == receipt["path"], "Extraction receipt path differs")
    receipt_path = root / Path(receipt["path"]).relative_to("/data/axonai")
    verify_file(receipt_path, receipt["sha256"])
    extraction = json.loads(receipt_path.read_bytes())
    check(extraction["record_count"] == start["train_examples"] and extraction["id_coverage_exact"] is True
          and extraction["approved_manifest_sha256"] == contract["manifest"]["sha256"] == identity["dataset_manifest_sha256"]
          and extraction["extracted_jsonl_sha256"] == identity["extracted_jsonl_sha256"] == bindings["train_sha256"]
          == approval["training_jsonl_sha256"] == provenance["dataset"]["sha256"], "Training input differs")
    print(f"Verified launch approval, {len(members)} snapshot members, extraction receipt and step-668 result.")


def merge_sources(root, provenance, external=None):
    package = root / "engineering-maintenance/merged-v1-packaging-v1-20261109"
    historical_package = Path("/data/axonai") / package.relative_to(root)
    artifacts_path = package / "ARTIFACT_MANIFEST.json"
    verify_file(artifacts_path, "69457aa564895f5a7bd59b9fd4895c7b8fe3d6d368ab6b66523aebad96cb1545")
    artifacts = json.loads(artifacts_path.read_bytes())
    rows = {row["path"]: row for row in artifacts["files"]}
    model = artifacts["model_manifest"]
    check(model["sha256"] == BF16_DIGEST and model["path"] == "/data/axonai/models/merged/AxonAI-v1-27B/" + MANIFEST,
          "Package names a different BF16 release")
    verify_file(BF16 / MANIFEST, model["sha256"], model["bytes"])
    missing = set()

    def verify_member(name, digest=None, size=None):
        check(Path(name).name == name and name not in ("", ".", ".."), "Invalid campaign filename")
        row = rows[str(historical_package / name)]
        check((digest is None or row["sha256"] == digest) and (size is None or row["bytes"] == size),
              f"Conflicting campaign records for {name}")
        path = package / name
        if name == "VALIDATION_PROMPTS.json" and not path.is_file():
            if external is None:
                missing.add(name)
                return
            path = external / name
        verify_file(path, row["sha256"], row["bytes"])

    for record in ("EXECUTION_FREEZE.json", "REPAIR_EXECUTION_FREEZE.json", "REPAIR2_EXECUTION_FREEZE.json"):
        verify_member(record)
        freeze = json.loads((package / record).read_bytes())
        for name, digest in freeze.items():
            verify_member(name, digest)
    for name in ("CPU_TEST_RESULTS.json", "VALIDATION_REPAIR.diff", "MERGE_RESULT.json", "preflight.py",
                 "compare.py", "summarize.py", "verify_equations.py", "package_support.py", "finalize_package.py"):
        verify_member(name)
    tests = json.loads((package / "CPU_TEST_RESULTS.json").read_bytes())
    check(sum(t["count"] for t in tests["tests"]) == tests["passed"] == 9 and tests["failed"] == 0,
          "Historical CPU test totals differ")
    for row in [t["file"] for t in tests["tests"]] + [tests["weight_equation_check"]]:
        verify_member(Path(row["path"]).name, row["sha256"], row["bytes"])
    preflight = json.loads((package / "PREFLIGHT.json").read_bytes())
    authorization = json.loads((package / "AUTHORIZATION.json").read_bytes())
    check(preflight["all_passed"] and preflight["verified_source_files"] == provenance["base_files"]
          and preflight["adapter"] == provenance["adapter"]
          and preflight["base_revision"] == authorization["base_revision"] == provenance["base_revision"]
          and authorization["adapter_sha256"] == provenance["adapter"]["sha256"],
          "Historical preflight or authorization differs from the BF16 release")
    check(json.loads((package / "MERGE_RESULT.json").read_bytes()) == provenance["merge"],
          "Archived merge result differs from release provenance")
    bf = manifest(BF16, BF16_DIGEST)
    summary = rows[str(historical_package / "VALIDATION_SUMMARY.json")]
    check(summary["sha256"] == bf["VALIDATION_SUMMARY.json"]["sha256"]
          and summary["bytes"] == bf["VALIDATION_SUMMARY.json"]["bytes"], "Release validation summary differs")
    verify_file(BF16 / "VALIDATION_SUMMARY.json", summary["sha256"], summary["bytes"])
    for name in ("MERGE_EQUATION_VERIFICATION.json", "standalone_inference.py", "verify_manifest.py"):
        verify_file(package / name, bf[name]["sha256"], bf[name]["bytes"])
    print("Verified BF16 campaign source and available records against the freezes and release provenance.")
    if missing:
        print("Recorded inputs requiring external bytes for a full hash check:", ", ".join(sorted(missing)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, help="directory corresponding to /data/axonai")
    parser.add_argument("--bf16-inputs", type=Path,
                        help="BF16 campaign directory containing the external frozen inputs; require all hashes")
    parser.add_argument("--check-input", action="append",
                        choices=("dataset", "attribution", "config", "adapter", "base"),
                        help="stream and hash this group of original inputs (repeatable)")
    args = parser.parse_args()
    if bool(args.archive_root) != bool(args.check_input):
        parser.error("supply --archive-root and --check-input together")
    evidence, register = release_records()
    recovered_sources(Path(__file__).resolve().parents[2], evidence)
    merge_sources(Path(__file__).resolve().parents[2], evidence, args.bf16_inputs)
    if args.check_input:
        archive_inputs(args.archive_root, args.check_input, evidence, register)
