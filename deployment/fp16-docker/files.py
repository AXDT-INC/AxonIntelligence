"""Filesystem helpers. All conversion writes are private to the output mount."""
import hashlib
import json
import os
from pathlib import Path

SOURCE = Path("/source")
OUTPUT = Path("/output")
EVIDENCE = OUTPUT / ".conversion"
WORK = EVIDENCE / "work"
MANIFEST = "SHA256_MANIFEST.json"
SOURCE_SHA = "d34c6389c0ca68df59e11f2a18f3616e7b335066fd8c493acf46044551e1daee"


def require(condition, message):
    if not condition:
        raise ValueError(str(message))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def source_rows():
    data = (SOURCE / MANIFEST).read_bytes()
    require(hashlib.sha256(data).hexdigest() == SOURCE_SHA, "Source manifest mismatch")
    rows = json.loads(data)["files"]
    names = [r["relative_path"] for r in rows]
    require(all(n == Path(n).name for n in names), "Invalid source pathname")
    require(len(names) == len(set(names)) == 33, "Source inventory mismatch")
    require(set(names) | {MANIFEST} == {p.name for p in SOURCE.iterdir()}, "Source inventory changed")
    return data, rows


def commit(staged, final):
    """Sync bytes, then link without replacing a previously committed file."""
    with staged.open("rb") as f:
        os.fsync(f.fileno())
    os.link(staged, final)
    fd = os.open(final.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    staged.unlink()


def publish(path, data):
    if path.exists():
        require(path.read_bytes() == data, f"Existing file differs: {path}")
        return
    temp = WORK / "record.partial"
    temp.unlink(missing_ok=True)
    with temp.open("xb") as stream:
        stream.write(data)
    commit(temp, path)
