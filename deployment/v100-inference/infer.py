"""Resident V100 inference and the reserved smoke run. CUDA imports are lazy."""
import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from placement import ANCHOR, INPUT, OUTPUT, plan, read_model
from score_cases import score

SYSTEM = ("You are AxonAI, an experimental scientific research assistant. "
          "State uncertainty and distinguish supplied evidence from inference.")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def uuids(text):
    values = text.split(",")
    pattern = r"GPU-[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}"
    require(len(values) in (2, 3, 4) and len(set(values)) == len(values)
            and all(re.fullmatch(pattern, v) for v in values), "Invalid reservation UUIDs")
    return values


def emit(**fields):
    print(json.dumps(fields, sort_keys=True), flush=True)


def greedy(model, ids, eos, limit=OUTPUT, *, max_input=INPUT, max_output=OUTPUT):
    """One unpadded sequence. Check every raw vocabulary logit before selecting a token."""
    import torch
    require(ids.shape[0] == 1 and 0 < ids.shape[1] <= max_input and 1 <= limit <= max_output, "Token budget")
    sequence = []
    cache = None
    start = time.perf_counter()
    for step in range(limit):
        result = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        logits = result.logits
        require(tuple(logits.shape) == (1, 1, model.config.vocab_size), "Raw logits shape")
        require(torch.isfinite(logits).all().item(), f"Nonfinite raw logits at generation step {step}")
        token = int(logits[0, 0].argmax().item())
        if step == 0:
            first = time.perf_counter() - start
        sequence.append(token)
        cache = result.past_key_values
        require(cache is not None, "Missing decode cache")
        if token in eos:
            break
        ids = ids.new_tensor([[token]])
    return sequence, cache, first, time.perf_counter() - start


def peaks(torch, count):
    for i in range(count):
        torch.cuda.synchronize(i)
    return [{"allocated_bytes": torch.cuda.max_memory_allocated(i),
             "reserved_bytes": torch.cuda.max_memory_reserved(i),
             "free_bytes_at_sample": torch.cuda.mem_get_info(i)[0]} for i in range(count)]


def load_resident(root, reserved, max_input, max_output):
    require(bool(os.statvfs(root).f_flag & os.ST_RDONLY), "Model mount must be read-only")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ",".join(reserved) ==
            os.environ.get("NVIDIA_VISIBLE_DEVICES"), "Reservation visibility mismatch")
    exposed = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).splitlines()
    require(len(exposed) == len(reserved) == 2 and set(exposed) == set(reserved),
            f"Unexpected GPUs visible to NVML: {exposed}")
    emit(stage="runtime_visibility", nvml_uuids=exposed)
    cfg, tensors = read_model(root)
    count = len(reserved)
    device_map, rows = plan(cfg, tensors, count, max_input, max_output)
    require(cfg["dtype"] == "float16" and cfg["mamba_ssm_dtype"] == "float32" and
            cfg["architectures"] == ["Qwen3_5ForCausalLM"], "Unexpected model config")
    for name in ("flash_attn", "fla", "causal_conv1d", "kernels"):
        require(importlib.util.find_spec(name) is None, f"Unexpected optional backend {name}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    require(torch.__version__ == "2.13.0+cu126" and torch.version.cuda == "12.6"
            and "sm_70" in torch.cuda.get_arch_list(), "Runtime lacks the selected Volta build")
    require(torch.cuda.device_count() == count, "Unexpected device count")
    for i, uuid in enumerate(reserved):
        props = torch.cuda.get_device_properties(i)
        require("V100" in props.name and (props.major, props.minor) == (7, 0), "Expected V100")
        require(str(props.uuid).lower().removeprefix("gpu-") == uuid[4:].lower(), f"UUID mismatch on {i}")
        require(torch.cuda.mem_get_info(i)[0] >= rows[i]["budget"], f"Insufficient free VRAM on {uuid}")
        torch.cuda.reset_peak_memory_stats(i)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    start = time.perf_counter()
    from safetensors import safe_open
    from unittest.mock import patch

    def open_checkpoint(filename, *args, **kwargs):
        if kwargs.get("device", "cpu") == "cpu":
            kwargs["backend"] = "pread"
        return safe_open(filename, *args, **kwargs)

    with patch("transformers.modeling_utils.safe_open", open_checkpoint):
        model, info = AutoModelForCausalLM.from_pretrained(
            root, local_files_only=True, trust_remote_code=False, dtype=torch.float16,
            device_map=device_map, attn_implementation="sdpa", use_kernels=False, output_loading_info=True)
    require(not any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys",
                                          "error_msgs", "conversion_errors")), f"Weight load report: {info}")
    model.eval().requires_grad_(False)
    params = dict(model.named_parameters())
    require(model.__class__.__name__ == "Qwen3_5ForCausalLM" and params.keys() == tensors.keys(), "Model identity")
    for key, tensor in params.items():
        group = ".".join(key.split(".")[:3]) if key.startswith("model.layers.") else key.rsplit(".", 1)[0]
        require(tensor.dtype == torch.float16 and tensor.numel()*2 == tensors[key] and
                tensor.device == torch.device("cuda", device_map[group]), f"Loaded parameter mismatch: {key}")
    load_peak = peaks(torch, count)
    emit(stage="loaded", seconds=time.perf_counter()-start, peak=load_peak, uuids=reserved,
         model_manifest_sha256=ANCHOR, placement=rows)
    eos = set(json.loads((root / "generation_config.json").read_text())["eos_token_id"])
    return Resident(model, tokenizer, cfg["layer_types"], eos, max_input, max_output)


class InvalidPrompt(ValueError):
    pass


@dataclass
class Resident:
    model: object
    tokenizer: object
    layer_types: list
    eos: set
    max_input: int
    max_output: int

    def encode(self, prompt):
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        if not 0 < ids.shape[1] <= self.max_input:
            raise InvalidPrompt(f"Rendered prompt exceeds {self.max_input} input tokens")
        return ids

    def decode(self, ids, limit):
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
            device = self.model.get_input_embeddings().weight.device
            tokens, cache, first, elapsed = greedy(self.model, ids.to(device), self.eos, limit,
                                                   max_input=self.max_input, max_output=self.max_output)
            for layer, kind in zip(cache.layers, self.layer_types, strict=True):
                if kind == "linear_attention":
                    state = layer.recurrent_states[0]
                    require(state.dtype == torch.float32 and torch.isfinite(state).all().item(), "DeltaNet state")
            text = self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return text, tokens, first, elapsed

    def generate(self, prompt, limit):
        ids = self.encode(prompt)
        text, tokens, _, _ = self.decode(ids, limit)
        return {"text": text, "input_tokens": ids.shape[1], "output_tokens": len(tokens)}


def run(root, reserved):
    resident = load_resident(root, reserved, INPUT, OUTPUT)
    import torch
    count = len(reserved)
    fixture = json.loads(Path(__file__).with_name("seven-prospective.json").read_text())
    outputs = {}
    smoke = {"id": "initial_decode", "prompt": "Count from one to five using digits."}
    for case in [smoke, *fixture]:
        ids = resident.encode(case["prompt"])
        for i in range(count):
            torch.cuda.reset_peak_memory_stats(i)
        initial = case is smoke
        emit(stage="starting_case", case=case["id"])
        continuation, tokens, first, elapsed = resident.decode(ids, 4 if initial else OUTPUT)
        emit(case=case["id"], input_tokens=ids.shape[1], generated_ids=tokens, seconds=elapsed,
             first_token_s=first, tokens_per_s=len(tokens)/elapsed, peak=peaks(torch,count),
             continuation=continuation, cached_decode_steps=len(tokens)-1, all_raw_logits_finite=True)
        if initial:
            require(len(tokens) >= 2, "Initial prompt ended before a cached decode step")
            emit(stage="initial_prefill_and_decode_passed")
        else:
            outputs[case["id"]] = continuation
    results = score(outputs)
    emit(suite="inherited prospective controls", passed=sum(results.values()), total=len(results),
         results=results, historical_replay=False)
    return 0 if all(results.values()) else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reserved-uuids", type=uuids, required=True)
    parser.add_argument("--model", type=Path, default=Path("/model"))
    args = parser.parse_args()
    try:
        raise SystemExit(run(args.model, args.reserved_uuids))
    except Exception as error:
        emit(stage="failed", error_type=type(error).__name__, error=str(error))
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            emit(stage="failure_allocator_peaks", devices=[
                {"allocated_bytes": torch.cuda.max_memory_allocated(i),
                 "reserved_bytes": torch.cuda.max_memory_reserved(i)}
                for i in range(torch.cuda.device_count())])
        raise
