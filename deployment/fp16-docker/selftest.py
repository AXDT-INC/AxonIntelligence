"""Tiny synthetic fixtures, including every finite BF16 input representable as finite FP16."""
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from safetensors.torch import save_file

from files import digest
from convert import check_staged, tensors_for_shard
from verify import compare_shard, rounding_table


def expect_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("Negative fixture was accepted")


def benchmark(root):
    table = rounding_table()
    size = 16 * 1024 * 1024
    words = np.resize(np.arange(65536, dtype="<u2")[table[1]], size)
    source = torch.from_numpy(words).view(torch.bfloat16)
    src, out = root / "benchmark-bf16", root / "benchmark-fp16"
    save_file({"weight": source}, src)
    save_file({"weight": source.to(torch.float16)}, out)
    start = perf_counter()
    counts, source_hash, output_hash = compare_shard(src, out, {"weight"}, table)
    elapsed = perf_counter() - start
    assert counts["elements"] == size and source_hash == digest(src) and output_hash == digest(out)
    print(f"Verifier benchmark: {size} values, {elapsed:.4f} s, {size/elapsed/1e6:.2f} million values/s "
          "(both payloads, SHA-256 and bitwise rounding checks)")


def test(root):
    src, out = root / "a", root / "b"
    reference = rounding_table()
    raw = np.arange(65536, dtype="<u2")[reference[1]]
    fixture = {"weights": torch.from_numpy(raw).view(torch.bfloat16), "bias": torch.tensor([-0., 1.]),
               "integer": torch.tensor([17, -42]), "scalar": torch.tensor(3.), "empty": torch.empty(0)}
    meta = {"format": "pt", "note": "synthetic"}
    keys = set(fixture)
    save_file(fixture, src, metadata=meta)
    values, metadata = tensors_for_shard(src, keys)
    save_file(values, out, metadata=metadata)
    check_staged(out, values, metadata)
    result, src_hash, out_hash = compare_shard(src, out, keys, reference)
    assert src_hash == digest(src) and out_hash == digest(out)
    assert result["converted"] == 1 and result["preserved"] == 4
    values["weights"][0] = 1
    save_file(values, out, metadata=meta)
    expect_error(lambda: compare_shard(src, out, keys, reference))
    for value in [torch.inf, -torch.inf, torch.nan, 65536.0]:
        save_file({"x": torch.tensor(value, dtype=torch.bfloat16)}, src)
        expect_error(lambda: tensors_for_shard(src, {"x"}))
    print("Synthetic CPU conversion, exhaustive finite rounding, preservation and rejection tests passed.")


if __name__ == "__main__":
    torch.set_num_threads(2)
    with TemporaryDirectory() as temp:
        test(Path(temp))
        benchmark(Path(temp))
