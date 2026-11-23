"""Independent verification using IEEE binary16 via struct, raw payloads and NumPy."""
import collections
import hashlib
import json
import math
import os
import struct
import time

import numpy as np
from safetensors import safe_open

from files import EVIDENCE, MANIFEST, OUTPUT, SOURCE, SOURCE_SHA, WORK, digest, encoded, publish, require, source_rows


def rounding_table():
    bits = np.zeros(65536, dtype="<u2")
    valid = np.zeros(65536, dtype=bool)
    for word in range(65536):
        value, = struct.unpack("<f", struct.pack("<I", word << 16))
        if math.isfinite(value):
            try:
                bits[word], = struct.unpack("<H", struct.pack("<e", value))
                valid[word] = True
            except OverflowError:
                pass
    return bits, valid


def read_header(stream):
    prefix = stream.read(8)
    length, = struct.unpack("<Q", prefix)
    require(0 < length < 100_000_000, "Invalid header size")
    raw = stream.read(length)
    return json.loads(raw), stream.tell(), hashlib.sha256(prefix + raw)


def compare_shard(src, out, expected_names, table):
    bits, valid = table
    counts = collections.Counter()
    with src.open("rb") as a, out.open("rb") as b, safe_open(out, framework="np") as reader:
        original, start_a, hash_a = read_header(a)
        written, start_b, hash_b = read_header(b)
        require(original.pop("__metadata__", None) == written.pop("__metadata__", None), f"Metadata: {src}")
        require(set(original) == set(written) == set(reader.keys()) == expected_names, f"Names: {src}")
        ordered = sorted(original, key=lambda n: original[n]["data_offsets"])
        dest_order = sorted(written, key=lambda n: written[n]["data_offsets"])
        same_order = ordered == dest_order
        for name in ordered:
            source_info = original[name]
            dest = written[name]
            shape, dtype = source_info["shape"], source_info["dtype"]
            converted = dtype == "BF16"
            require(shape == dest["shape"] == reader.get_slice(name).get_shape(), name)
            require(dest["dtype"] == ("F16" if converted else dtype), name)
            left, right = source_info["data_offsets"]
            lout, rout = dest["data_offsets"]
            count = math.prod(shape)
            require(right - left == rout - lout, name)
            if converted:
                require(right - left == 2 * count, name)
            require(a.tell() == start_a + left, f"Source offsets: {name}")
            if same_order:
                require(b.tell() == start_b + lout, f"Output offsets: {name}")
            else:
                b.seek(start_b + lout)
            remaining = right - left
            while remaining:
                size = min(remaining, 4 << 20)
                x, y = a.read(size), b.read(size)
                require(len(x) == len(y) == size, f"Truncated payload: {name}")
                hash_a.update(x)
                if same_order:
                    hash_b.update(y)
                if converted:
                    words = np.frombuffer(x, dtype="<u2")
                    require(valid[words].all(), f"Nonfinite or overflowing source: {name}")
                    require(np.array_equal(bits[words], np.frombuffer(y, dtype="<u2")),
                            f"Rounding mismatch in {name} at byte {right-left-remaining}")
                else:
                    require(x == y, f"Preserved bytes differ: {name}")
                    floats = {"F16": "<f2", "F32": "<f4", "F64": "<f8", "C64": "<c8", "C128": "<c16"}
                    if dtype in floats:
                        require(np.isfinite(np.frombuffer(y, dtype=floats[dtype])).all(), name)
                remaining -= size
            counts["elements"] += count
            counts["weight_bytes"] += right - left
            counts["converted" if converted else "preserved"] += 1
        require(not a.read(1), f"Trailing bytes: {src}")
        if same_order:
            require(not b.read(1), f"Trailing bytes: {out}")
    return counts, hash_a.hexdigest(), hash_b.hexdigest() if same_order else digest(out)


def verify():
    source_manifest, rows = source_rows()
    require((EVIDENCE / "source-manifest.json").read_bytes() == source_manifest, "Source evidence mismatch")
    names = {row["relative_path"] for row in rows}
    expected_root = names | {EVIDENCE.name}
    require({p.name for p in OUTPUT.iterdir()} - {MANIFEST} == expected_root, "Output inventory")
    idx = json.loads((SOURCE / "model.safetensors.index.json").read_bytes())
    mapping = idx["weight_map"]
    shards = {n for n in names if n.endswith(".safetensors")}
    require(len(shards) == 14 and set(mapping.values()) == shards, "Index references")
    expected_records = {name + ".json" for name in shards}
    require({p.name for p in (EVIDENCE / "shards").iterdir()} == expected_records, "Shard records")
    require({p.name for p in EVIDENCE.iterdir()} - {"verification.json"} ==
            {"source-manifest.json", "work", "shards"}, "Unexpected conversion evidence")
    require(not any(WORK.iterdir()), "Incomplete staging files")
    table = rounding_table()
    totals = collections.Counter()
    entries = []
    seconds = 0.0
    for row in rows:
        name = row["relative_path"]
        a, b = SOURCE / name, OUTPUT / name
        if name in shards:
            counts, original_sha, checksum = compare_shard(a, b, {k for k, v in mapping.items() if v == name}, table)
            totals.update(counts)
        else:
            original_sha, checksum = digest(a), digest(b)
        require(original_sha == row["sha256"] and a.stat().st_size == row["bytes"], f"Source hash: {name}")
        if name in shards:
            record = json.loads((EVIDENCE / "shards" / (name + ".json")).read_bytes())
            require(record["sha256"] == checksum and record["name"] == name and
                    record["source_sha256"] == row["sha256"] and record["bytes"] == b.stat().st_size and
                    record["tensors"] == sum(v == name for v in mapping.values()), f"Shard evidence: {name}")
            seconds += record["seconds"]
            print(f"Independently verified {name}", flush=True)
        elif name == "config.json":
            original, output = json.loads(a.read_bytes()), json.loads(b.read_bytes())
            require(original.pop("dtype") == "bfloat16" and output.pop("dtype") == "float16" and
                    original == output and output["architectures"] == ["Qwen3_5ForCausalLM"], "Model config")
        else:
            require(checksum == row["sha256"], f"Supporting file: {name}")
        entries.append({"relative_path": name, "bytes": b.stat().st_size, "sha256": checksum})
    require(totals["elements"] == idx["metadata"]["total_parameters"] == 26_895_998_464 and
            totals["weight_bytes"] == idx["metadata"]["total_size"] and
            totals["converted"] + totals["preserved"] == len(mapping) == 851, "Model totals")
    report = dict(totals, status="passed", shards=len(shards), source_manifest_sha256=SOURCE_SHA,
                  method="Exhaustive BF16 lookup using Python struct IEEE binary16, bitwise payload comparison",
                  shard_conversion_seconds=round(seconds, 3))
    publish(EVIDENCE / "verification.json", encoded(report))
    for path in sorted(EVIDENCE.rglob("*")):
        if path.is_file():
            entries.append({"relative_path": str(path.relative_to(OUTPUT)), "bytes": path.stat().st_size,
                            "sha256": digest(path)})
    manifest = encoded({"schema": "axonai.standalone-model-sha256.v1", "files": entries,
                        "payload_bytes": sum(e["bytes"] for e in entries), "self_hash_excluded": True})
    publish(OUTPUT / MANIFEST, manifest)
    print(json.dumps(dict(report, output_bytes=sum(e["bytes"] for e in entries) + len(manifest),
                          manifest_sha256=hashlib.sha256(manifest).hexdigest()), indent=2), flush=True)


if __name__ == "__main__":
    os.umask(0o077)
    started = time.monotonic()
    verify()
    print(f"Independent verification finished in {time.monotonic() - started:.3f} seconds", flush=True)
