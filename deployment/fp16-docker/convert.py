import fcntl
import hashlib
import json
import os
import time

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

from files import EVIDENCE, MANIFEST, OUTPUT, SOURCE, WORK, commit, digest, encoded, publish, require, source_rows


def check_values(tensor, name):
    if tensor.is_floating_point() or tensor.is_complex():
        for part in tensor.reshape(-1).split(2**20):
            require(torch.isfinite(part).all().item(), f"Nonfinite input or FP16 overflow: {name}")


def tensors_for_shard(path, keys):
    with safe_open(path, framework="pt", device="cpu") as src:
        require(set(src.keys()) == keys, f"Index mismatch: {path}")
        result = {}
        for name in src.keys():
            original = src.get_tensor(name)
            value = original.to(torch.float16) if original.dtype == torch.bfloat16 else original
            # A BF16 NaN, infinity or overflowing finite value stays nonfinite after the cast.
            check_values(value, name)
            result[name] = value
        return result, src.metadata()


def check_staged(path, values, metadata):
    with safe_open(path, framework="pt", device="cpu") as out:
        require(out.metadata() == metadata and set(out.keys()) == set(values), path)
        for name, expected in values.items():
            actual = out.get_tensor(name)
            require(actual.dtype == expected.dtype and actual.shape == expected.shape, name)
            require(torch.equal(actual.reshape(-1).view(torch.uint8),
                                expected.reshape(-1).view(torch.uint8)), f"Readback mismatch: {name}")


def convert_one(row, keys, record):
    name = row["relative_path"]
    staged = WORK / (name + ".partial")
    started = time.monotonic()
    values, metadata = tensors_for_shard(SOURCE / name, keys)
    if staged.exists():
        # An interrupted save can leave a complete or truncated staging file.
        try:
            check_staged(staged, values, metadata)
        except (ValueError, RuntimeError, OSError, SafetensorError):
            staged.unlink()
    if not staged.exists():
        save_file(values, staged, metadata=metadata)
        check_staged(staged, values, metadata)
    del values
    receipt = {"name": name, "source_sha256": row["sha256"], "sha256": digest(staged),
               "bytes": staged.stat().st_size, "tensors": len(keys),
               "seconds": round(time.monotonic() - started, 3)}
    publish(record, encoded(receipt))
    commit(staged, OUTPUT / name)
    print(f"Verified and finalized {name}  {receipt['sha256']}", flush=True)


def convert():
    data, rows = source_rows()
    expected = {r["relative_path"] for r in rows}
    shards = [r for r in rows if r["relative_path"].endswith(".safetensors")]
    index = json.loads((SOURCE / "model.safetensors.index.json").read_bytes())["weight_map"]
    require(len(shards) == 14 and set(index.values()) == {r["relative_path"] for r in shards}, "Shard set")
    existing = {p.name for p in OUTPUT.iterdir()}
    require(not existing or (EVIDENCE.is_dir() and existing <= expected | {EVIDENCE.name, MANIFEST}),
            "Destination contains unrelated files")
    if existing:
        for row in rows:
            require(digest(SOURCE / row["relative_path"]) == row["sha256"], row["relative_path"])
    WORK.mkdir(parents=True, exist_ok=True)
    records = EVIDENCE / "shards"
    records.mkdir(exist_ok=True)
    publish(EVIDENCE / "source-manifest.json", data)
    completed = set()
    for row in shards:
        name = row["relative_path"]
        final, record = OUTPUT / name, records / (name + ".json")
        staged = WORK / (name + ".partial")
        if not record.exists():
            require(not final.exists(), f"Unrecorded final shard: {name}")
            continue
        info = json.loads(record.read_bytes())
        ready = final if final.exists() else staged
        require(info["name"] == name and info["source_sha256"] == row["sha256"] and
                ready.stat().st_size == info["bytes"] and digest(ready) == info["sha256"], name)
        if not final.exists():
            commit(staged, final)
        elif staged.exists() and staged.samefile(final):
            staged.unlink()
        completed.add(name)
    for row in shards:
        name = row["relative_path"]
        if name not in completed:
            convert_one(row, {k for k, v in index.items() if v == name}, records / (name + ".json"))
    for row in rows:
        name = row["relative_path"]
        if name.endswith(".safetensors"):
            continue
        original = (SOURCE / name).read_bytes()
        require(hashlib.sha256(original).hexdigest() == row["sha256"], name)
        if name == "config.json":
            cfg = json.loads(original)
            require(cfg["dtype"] == "bfloat16", "Unexpected source dtype declaration")
            cfg["dtype"] = "float16"
            original = (json.dumps(cfg, indent=2) + "\n").encode()
        publish(OUTPUT / name, original)
    # Recover the small-file temporary link if interrupted just after link(2).
    temp = WORK / "record.partial"
    if temp.exists():
        temp.unlink()


if __name__ == "__main__":
    os.umask(0o077)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    fd = os.open(OUTPUT, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic()
        with torch.inference_mode():
            convert()
        print(f"Conversion invocation finished in {time.monotonic() - start:.3f} seconds", flush=True)
    finally:
        os.close(fd)
