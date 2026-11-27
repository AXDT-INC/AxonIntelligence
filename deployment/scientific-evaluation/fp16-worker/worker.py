"""One offline FP16 worker per checkpoint. Emit JSON Lines on stdout."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from dataclasses import replace
from unittest.mock import patch

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from infer import Resident, peaks, require, uuids
from placement import ANCHOR, plan

MODEL = Path("/model")
PINS = {"axonai": ANCHOR, "qwen": "644bca802d87721db6656d3ef4f7bf8e8c79bfdbe2761c4abce5086186b06c6e"}
SYSTEM = "You are a scientific research assistant. Follow the requested answer format."
INPUT = 2048
OUTPUT = 256
PROFILES = {
    "v100": {"name": "V100", "capability": (7, 0), "torch": "2.13.0+cu126", "cuda": "12.6", "arch": "sm_70"},
    "l40s": {"name": "NVIDIA L40S", "capability": (8, 9), "torch": "2.13.0+cu132", "cuda": "13.2", "arch": "sm_86"},
}


def hardware_profile():
    name = os.environ.get("BENCH_GPU_PROFILE", "v100")
    require(name in PROFILES, "Unknown hardware profile")
    return name, PROFILES[name]


def emit(**fields):
    print(json.dumps(fields, sort_keys=True, ensure_ascii=False, allow_nan=False), flush=True)


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()



def network_isolated():
    names = {name for _, name in socket.if_nameindex()}
    if "lo" not in names or not names <= {"lo", "tunl0"}:
        return False
    # Some host kernels create a dormant IPIP fallback device in every new netns.
    # It is acceptable only while administratively down, unaddressed, and unrouted.
    if "tunl0" in names and int(Path("/sys/class/net/tunl0/flags").read_text().strip(), 16) & 1:
        return False
    for filename in ("/proc/net/route", "/proc/net/ipv6_route"):
        path = Path(filename)
        if path.exists():
            lines = path.read_text().splitlines()
            if any("tunl0" in line.split() for line in lines):
                return False
    addresses = Path("/proc/net/if_inet6")
    if addresses.exists() and any(line.split()[-1] != "lo" for line in addresses.read_text().splitlines()):
        return False
    if "tunl0" in names:
        import errno
        import fcntl
        import struct
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as control:
            try:
                fcntl.ioctl(control.fileno(), 0x8915, struct.pack("256s", b"tunl0"))
            except OSError as error:
                if error.errno != errno.EADDRNOTAVAIL:
                    return False
            else:
                return False
    return True


def isolation(device):
    require(os.getuid() != 0 and os.statvfs("/").f_flag & os.ST_RDONLY and
            os.statvfs(MODEL).f_flag & os.ST_RDONLY, "Expected unprivileged read-only mounts")
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    require(int(status["CapBnd"], 16) == 0 and int(status["NoNewPrivs"]) == 1, "Worker privileges")
    require(network_isolated() and
            not Path("/var/run/docker.sock").exists(), "Worker isolation")
    if device == "cpu":
        require(not list(Path("/dev").glob("nvidia*")) and torch.cuda.device_count() == 0,
                "Unexpected NVIDIA devices in CPU qualification")


def checkpoint(arm):
    pin = PINS[arm]
    manifest = (MODEL / "SHA256_MANIFEST.json").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != pin:
        raise ValueError("Benchmark manifest digest mismatch")
    records = {row["relative_path"]: row for row in json.loads(manifest)["files"]}
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json",
                 "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        require(hashlib.sha256((MODEL / name).read_bytes()).hexdigest() == records[name]["sha256"], name)
    cfg = json.loads((MODEL / "config.json").read_bytes())
    sizes = {}
    index = json.loads((MODEL / "model.safetensors.index.json").read_bytes())
    shards = sorted(set(index["weight_map"].values()))
    require(len(shards) == 14 and {p.name for p in MODEL.glob("*.safetensors")} == set(shards)
            and not (MODEL / "adapter_config.json").exists(), "Unexpected model or adapter inventory")
    for name in shards:
        path = MODEL / name
        require(path.stat().st_size == records[name]["bytes"], name)
        with path.open("rb") as f:
            require(hashlib.file_digest(f, "sha256").hexdigest() == records[name]["sha256"], name)
        with safe_open(path, framework="pt", device="cpu", backend="pread") as shard:
            require(set(shard.keys()) == {k for k, v in index["weight_map"].items() if v == name}, name)
            for key in shard.keys():
                item = shard.get_slice(key)
                require(item.get_dtype() == "F16", key)
                sizes[key.replace("model.language_model.", "model.", 1)] = 2 * math.prod(item.get_shape())
    require(len(sizes) == 851 and sum(sizes.values()) == 53_791_996_928, "Text checkpoint inventory")
    require(cfg["dtype"] == "float16" and cfg["mamba_ssm_dtype"] == "float32"
            and cfg["architectures"] == ["Qwen3_5ForCausalLM"], "Model config")
    return cfg, sizes


def gpu_pair(rows):
    reserved = uuids(os.environ.get("BENCH_RESERVED_UUIDS", ""))
    require(len(reserved) == 2 and os.environ["CUDA_VISIBLE_DEVICES"] == ",".join(reserved)
            == os.environ["NVIDIA_VISIBLE_DEVICES"], "Expected exactly the reserved UUIDs")
    exposed = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True)
    require(set(exposed.splitlines()) == set(reserved) and torch.cuda.device_count() == 2,
            "GPU visibility differs from reservation")
    _, profile = hardware_profile()
    for i, uuid in enumerate(reserved):
        props = torch.cuda.get_device_properties(i)
        require(profile["name"] in props.name and (props.major, props.minor) == profile["capability"]
                and str(props.uuid).lower().removeprefix("gpu-") == uuid[4:].lower(), "GPU identity or order")
        require(torch.cuda.mem_get_info(i)[0] >= rows[i]["budget"], "Insufficient free VRAM")
        torch.cuda.reset_peak_memory_stats(i)
    return reserved


def load(arm, device):
    isolation(device)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    cfg, sizes = checkpoint(arm)
    mapping, rows = plan(cfg, sizes, 2, INPUT, OUTPUT)
    ids = gpu_pair(rows) if device == "cuda" else []
    profile_name, profile = hardware_profile()
    require(torch.__version__ == profile["torch"] and torch.version.cuda == profile["cuda"]
            and profile["arch"] in torch._C._cuda_getArchFlags().split(), "Expected profile-pinned CUDA wheel")
    emit(event="verified", arm=arm, manifest=PINS[arm], tensors=len(sizes), weight_bytes=sum(sizes.values()),
         placement=rows, device_map=mapping, gpu_uuids=ids, device=device, hardware_profile=profile_name,
         runtime={"torch": torch.__version__, "cuda": torch.version.cuda,
                  "transformers": __import__("transformers").__version__,
                  "safetensors": __import__("safetensors").__version__})
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)

    def pread(filename, *args, **kwargs):
        if kwargs.get("device", "cpu") == "cpu":
            kwargs["backend"] = "pread"
        return safe_open(filename, *args, **kwargs)

    start = time.monotonic()
    with patch("transformers.modeling_utils.safe_open", pread):
        model, report = AutoModelForCausalLM.from_pretrained(
            MODEL, local_files_only=True, trust_remote_code=False, dtype=torch.float16,
            device_map=mapping if ids else {"": "cpu"}, attn_implementation="sdpa",
            use_kernels=False, output_loading_info=True)
    require(not any(report.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys",
                                             "error_msgs", "conversion_errors")), "Checkpoint load report")
    model.eval().requires_grad_(False)
    params = dict(model.named_parameters())
    require(model.__class__.__name__ == "Qwen3_5ForCausalLM" and params.keys() == sizes.keys(),
            "Text model identity")
    for key, tensor in params.items():
        group = ".".join(key.split(".")[:3]) if key.startswith("model.layers.") else key.rsplit(".", 1)[0]
        placement = torch.device("cuda", mapping[group]) if ids else torch.device("cpu")
        require(tensor.dtype == torch.float16 and tensor.device == placement
                and tensor.numel() * 2 == sizes[key], key)
    eos = set(json.loads((MODEL / "generation_config.json").read_bytes())["eos_token_id"])
    emit(event="loaded", arm=arm, elapsed_s=time.monotonic()-start, gpu=peaks(torch, len(ids)))
    return Resident(model, tokenizer, cfg["layer_types"], eos, INPUT, OUTPUT)


def encode(tokenizer, prompt):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
                                         return_dict=False, return_tensors="pt")


def response(resident, job):
    require(isinstance(job, dict) and set(job) == {"id", "prompt", "max_new_tokens"}, "Invalid job fields")
    ident, prompt, cap = job["id"], job["prompt"], job["max_new_tokens"]
    require(isinstance(ident, str) and 0 < len(ident) <= 128 and isinstance(prompt, str)
            and bool(prompt.strip()) and type(cap) is int and 1 <= cap <= OUTPUT, "Invalid job")
    if "<|im_start|>" in prompt or "<|im_end|>" in prompt:
        return {"id": ident, "status": "invalid_prompt"}
    ids = encode(resident.tokenizer, prompt)
    n = ids.shape[1]
    metadata = {"id": ident, "input_tokens": n, "input_ids_sha256": digest(ids[0].tolist())}
    if n > INPUT:
        return dict(metadata, status="context_overflow")
    cuda = resident.model.get_input_embeddings().weight.is_cuda
    if cuda:
        for i in range(2):
            torch.cuda.reset_peak_memory_stats(i)
    text, tokens, first, elapsed = resident.decode(ids, cap)
    return dict(metadata, status="ok" if text.strip() else "empty_output", text=text, generated_ids=tokens,
                output_tokens=len(tokens), stop="eos" if tokens[-1] in resident.eos else "cap",
                first_token_s=first, elapsed_s=elapsed, gpu=peaks(torch, 2) if cuda else [])


def qualify(resident, device, arm):
    job = {"id": "short", "prompt": "Write the digits from one to nine in order, one digit per line.",
           "max_new_tokens": 4}
    a, b = response(resident, job), response(resident, job)
    require(a["status"] == b["status"] == "ok" and a["generated_ids"] == b["generated_ids"]
            and a["input_ids_sha256"] == b["input_ids_sha256"] and a["output_tokens"] >= 2,
            "Greedy replay or cached decode failed")
    emit(event="replay", arm=arm, first=a, repeat=b)
    # For this tokenizer each repeated ' x' adds exactly one token.
    prefix = "Reply with the word blue."
    prompt = prefix + " x" * (INPUT - encode(resident.tokenizer, prefix).shape[1])
    ids = encode(resident.tokenizer, prompt)
    require(ids.shape[1] == INPUT, "Synthetic boundary fixture has wrong length")
    overflow = response(resident, dict(job, prompt=prompt+" x"))
    require(overflow["status"] == "context_overflow" and overflow["input_tokens"] == INPUT+1, "Input cap")
    boundary = {"input_tokens": INPUT, "input_ids_sha256": digest(ids[0].tolist()),
                "overlength_rejected": overflow["input_tokens"]}
    if device == "cuda":
        # Disabling EOS only for this synthetic stress probe exercises all 256 cache steps.
        stress = response(replace(resident, eos=set()), dict(job, prompt=prompt, max_new_tokens=OUTPUT))
        require(stress["stop"] == "cap" and stress["output_tokens"] == OUTPUT, "Full envelope")
        boundary.update(stress_output_tokens=stress["output_tokens"], stress_gpu=stress["gpu"],
                        stress_elapsed_s=stress["elapsed_s"])
    emit(event="qualified", arm=arm, device=device, manifest=PINS[arm], **boundary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=sorted(PINS))
    parser.add_argument("mode", choices=["qualify", "stream"], default="qualify", nargs="?")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    resident = load(args.arm, args.device)
    if args.mode == "qualify":
        qualify(resident, args.device, args.arm)
        return
    seen = set()
    for line in sys.stdin:
        job = json.loads(line)
        require(isinstance(job, dict) and isinstance(job.get("id"), str) and job["id"] not in seen,
                "Missing or repeated job ID")
        seen.add(job["id"])
        emit(event="outcome", arm=args.arm, manifest=PINS[args.arm], **response(resident, job))


if __name__ == "__main__":
    main()
