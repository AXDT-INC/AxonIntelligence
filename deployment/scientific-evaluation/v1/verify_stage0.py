"""Verify the fixed Stage 0 transfer offline. Never extract files or follow source references."""
import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path

PACKAGE = "axonai-v1-stage0-evaluation-evidence-20261124"
MANIFEST = "7b7acebb18f5747e434d1fc518256b1ba38cb117836791227a9e1b1f7b8ff739"
ARCHIVE = "6d14709bf22595325bf4dbcf2d72f006e1ad9a5b49bc925f744cb4967984bade"
EXP = "engineering-maintenance/exploratory-base-adapter-v1-20261030/"
CHECK = "engineering-maintenance/post-training-evaluation-v1-20261030/"
FILES = {"REPORT.md", *(EXP + name + ".v1.json" for name in
         ("CONCLUSION", "DESCRIPTIVE_SUMMARY", "PREGENERATION_LOCK", "RUBRIC")),
         CHECK + "COMPARISON_RESULTS.v1.json", CHECK + "ISOLATION_INCIDENT.v1.json"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def verify(parent):
    root = parent / PACKAGE
    archive = parent / (PACKAGE + ".tar.gz")
    manifest = root / "SHA256SUMS"
    require(root.is_dir() and not root.is_symlink(), "invalid package directory")
    require(all(p.is_file() and not p.is_symlink() for p in (archive, manifest)), "invalid transfer")
    require(digest(archive.read_bytes()) == ARCHIVE, "bundle digest mismatch")
    manifest_bytes = manifest.read_bytes()
    require(digest(manifest_bytes) == MANIFEST, "manifest digest mismatch")
    expected = {}
    for line in manifest_bytes.decode("ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match and match[2] in FILES and match[2] not in expected, "bad manifest entry")
        expected[match[2]] = match[1]
    require(expected.keys() == FILES, "incomplete manifest")
    expected["SHA256SUMS"] = MANIFEST
    actual = set()
    for path in root.rglob("*"):
        require(not path.is_symlink(), "link in package")
        if not path.is_dir():
            name = path.relative_to(root).as_posix()
            require(name in expected and path.is_file(), "unexpected unpacked member")
            require(digest(path.read_bytes()) == expected[name], "unpacked member digest mismatch")
            actual.add(name)
    require(actual == expected.keys(), "incomplete directory")
    seen = set()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            if member.isdir():
                continue
            prefix = PACKAGE + "/"
            require(member.isfile() and member.name.startswith(prefix), "unexpected archive entry")
            name = member.name[len(prefix):]
            require(name in expected and name not in seen, "unexpected archive member")
            require(digest(bundle.extractfile(member).read()) == expected[name], "archive member digest mismatch")
            seen.add(name)
    require(seen == expected.keys(), "incomplete archive")

    # Only project-level descriptive fields leave the private evidence directory.
    summary = json.loads((root / (EXP + "DESCRIPTIVE_SUMMARY.v1.json")).read_bytes())
    comparison = json.loads((root / (CHECK + "COMPARISON_RESULTS.v1.json")).read_bytes())
    counts = summary["pair_preference_counts"]
    return {
        "package": PACKAGE, "manifest_sha256": MANIFEST, "bundle_sha256": ARCHIVE,
        "payloads_verified_in_directory_and_archive": len(FILES),
        "paired_items": sum(counts.values()), "adapter_preferred": counts["finetuned_preferred"],
        "base_preferred": counts["base_preferred"], "ties": counts["no_clear_difference"],
        "expert_adjudication": summary["independent_expert_adjudication"],
        "review_type": summary["review_type"], "checkpoint_outputs": comparison["outputs_generated"],
        "checkpoint_scores": comparison["scores_assigned"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transfer_parent", type=Path)
    print(json.dumps(verify(parser.parse_args().transfer_parent), indent=2))
