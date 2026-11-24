"""Build reproducible source-only L40S transfer archive or check an unpacked private copy."""
import argparse
import hashlib
import io
from pathlib import Path
import tarfile

from runner import PROVENANCE_SHA256, require

HERE = Path(__file__).resolve().parent
FILES = ("runner.py", "live_backend.py", "live_worker.py", "live_store.py", "live_supervisor.py",
         "live_package.py", "PROVENANCE.json", "LIVE.md")
MANIFEST = "SHA256SUMS"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sums(blobs):
    return "".join(f"{sha(blobs[n])}  {n}\n" for n in FILES).encode()


def build(output):
    blobs = {n: (HERE / n).read_bytes() for n in FILES if n != "PROVENANCE.json"}
    blobs["PROVENANCE.json"] = (HERE.parents[2] / "models/merged/AxonAI-v1-27B/PROVENANCE.json").read_bytes()
    require(sha(blobs["PROVENANCE.json"]) == PROVENANCE_SHA256, "wrong provenance")
    blobs[MANIFEST] = sums(blobs)
    with output.open("xb") as dest, tarfile.open(fileobj=dest, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name in (*FILES, MANIFEST):
            info = tarfile.TarInfo(name)
            info.mode = 0o444
            info.size = len(blobs[name])
            archive.addfile(info, io.BytesIO(blobs[name]))
    return {"archive_sha256": sha(output.read_bytes()), "manifest_sha256": sha(blobs[MANIFEST])}


def verify(root, manifest_sha256):
    names = (*FILES, MANIFEST)
    require(set(p.name for p in root.iterdir()) == set(names) and all(
        (root/n).is_file() and not (root/n).is_symlink() for n in names), "wrong package inventory")
    blobs = {n: (root/n).read_bytes() for n in names}
    require(sha(blobs[MANIFEST]) == manifest_sha256 and blobs[MANIFEST] == sums(blobs)
            and sha(blobs["PROVENANCE.json"]) == PROVENANCE_SHA256, "package digest mismatch")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").add_argument("output", type=Path)
    check = sub.add_parser("verify")
    check.add_argument("directory", type=Path)
    check.add_argument("manifest_sha256")
    args = p.parse_args()
    if args.cmd == "build":
        print(build(args.output))
    else:
        verify(args.directory, args.manifest_sha256)
        print("Package verified.")
