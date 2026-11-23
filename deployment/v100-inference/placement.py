"""Batch-one memory plan using only checkpoint headers and the authenticated index."""
import hashlib
import itertools
import json
import math
import struct
from pathlib import Path

GIB = 1 << 30
ANCHOR = "c3a6cb1deb5123cc35ccc9c07de36545f82b91a317f97971b6bd56b3403f2e06"
INPUT = 256
OUTPUT = 32


def read_model(root):
    data = (root / "SHA256_MANIFEST.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != ANCHOR:
        raise ValueError("FP16 manifest digest mismatch")
    files = {r["relative_path"]: r for r in json.loads(data)["files"]}
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json",
                 "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != files[name]["sha256"]:
            raise ValueError(name)
    cfg = json.loads((root / "config.json").read_bytes())
    index = json.loads((root / "model.safetensors.index.json").read_bytes())
    mapping = index["weight_map"]
    sizes = {}
    for name in sorted(set(mapping.values())):
        path = root / name
        if path.stat().st_size != files[name]["bytes"]:
            raise ValueError(name)
        with path.open("rb") as stream:
            length, = struct.unpack("<Q", stream.read(8))
            if length > 10**7:
                raise ValueError(name)
            header = json.loads(stream.read(length))
        header.pop("__metadata__", None)
        if set(header) != {k for k, v in mapping.items() if v == name}:
            raise ValueError(f"Index mismatch in {name}")
        for key, h in header.items():
            size = math.prod(h["shape"]) * 2
            if h["dtype"] != "F16" or h["data_offsets"][1] - h["data_offsets"][0] != size:
                raise ValueError(key)
            sizes[key.replace("model.language_model.", "model.", 1)] = size
    if len(sizes) != 851 or sum(sizes.values()) != 53_791_996_928:
        raise ValueError("Unexpected checkpoint totals")
    return cfg, sizes


def plan(cfg, sizes, devices):
    if devices not in (2, 3, 4):
        raise ValueError("Expected 2, 3 or 4 devices")
    layers = cfg["layer_types"]
    n = len(layers)
    blocks = [sum(v for k, v in sizes.items() if k.startswith(f"model.layers.{i}.")) for i in range(n)]
    blocks[0] += sizes["model.embed_tokens.weight"]
    blocks[-1] += sizes["model.norm.weight"] + sizes["lm_head.weight"]
    if not all(blocks) or sum(blocks) != sum(sizes.values()):
        raise ValueError("Unassigned weight")
    prefix = [0, *itertools.accumulate(blocks)]
    def usage(cuts):
        return [prefix[b] - prefix[a] for a, b in zip((0, *cuts), (*cuts, n))]
    cuts = min(itertools.combinations(range(1, n), devices-1), key=lambda c: max(usage(c)))
    ranges = list(zip((0, *cuts), (*cuts, n)))
    device_map = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": devices-1, "lm_head": devices-1}
    # Per layer at batch one. Compute types follow the pinned native fallback.
    kv = 2 * cfg["num_key_value_heads"] * cfg["head_dim"] * (INPUT + OUTPUT) * 2
    recurrent = cfg["linear_num_value_heads"] * cfg["linear_key_head_dim"] * cfg["linear_value_head_dim"] * 4
    channels = 2*cfg["linear_num_key_heads"]*cfg["linear_key_head_dim"] + cfg["linear_num_value_heads"]*cfg["linear_value_head_dim"]
    conv = channels * cfg["linear_conv_kernel_dim"] * 2
    rows = []
    for dev, ((start, stop), weights) in enumerate(zip(ranges, usage(cuts))):
        for layer in range(start, stop):
            device_map[f"model.layers.{layer}"] = dev
        full = layers[start:stop].count("full_attention")
        linear = stop-start-full
        cache = full*kv + linear*(recurrent+conv)
        # Planning allowances, to be checked by measurement on the approved hardware.
        budget = weights + cache + 4*GIB
        if weights > 27*GIB:
            raise ValueError("Placement exceeds 27 GiB")
        rows.append(dict(device=dev, first=start, last=stop-1, weights=weights, kv=full*kv,
                         recurrent=linear*recurrent, conv=linear*conv, budget=budget))
    return device_map, rows


if __name__ == "__main__":
    import sys
    config, tensors = read_model(Path(sys.argv[1]))
    print(json.dumps({n:plan(config, tensors, n)[1] for n in (2,3,4)}, indent=2))
