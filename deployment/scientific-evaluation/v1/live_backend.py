"""Offline L40S implementation of runner.Decode. ML imports are lazy."""
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

import runner as r

BASE = Path("/model/base")
ADAPTER = Path("/model/adapter")
PROVENANCE = Path(__file__).with_name("PROVENANCE.json")
MODES = {"discover", "bare", "disabled", "adapter"}
PACKAGES = ("torch", "transformers", "peft", "safetensors", "tokenizers", "accelerate",
            "numpy", "huggingface-hub")
VOCAB = 248320


def placement():
    # Text-only 64-layer architecture. Keep rotary state and embeddings on the first device.
    layout = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": 1, "lm_head": 1}
    layout.update({f"model.layers.{i}": int(i >= 32) for i in range(64)})
    return layout


def device_for(name, layout):
    while name not in layout:
        r.require("." in name, "unmapped model parameter")
        name = name.rsplit(".", 1)[0]
    return layout[name]


def check_options(options):
    r.require(type(options) is list and len(options) <= 16, "invalid option inventory")
    for ids in options:
        r.require(type(ids) is list and 0 < len(ids) <= 32 and
                  all(type(t) is int and 0 <= t < VOCAB for t in ids), "invalid option tokens")
    r.require(len(set(map(tuple, options))) == len(options), "duplicate option")


def conditional_scores(prefix, options, next_logprobs):
    """Teacher-forced sum of conditional log probabilities; no label, length prior or EOS added."""
    import math
    check_options(options)
    rows = []
    for option in options:
        r.require(len(prefix) + len(option) <= r.CONTEXT, "option exceeds context")
        values = next_logprobs(prefix, option)
        if len(values) != len(option) or not all(math.isfinite(x) for x in values):
            raise r.ItemFailure("nonfinite_logits")
        rows.append({"tokens": len(values), "log_probability": math.fsum(values)})
    return rows


def smi(query):
    return subprocess.check_output(["nvidia-smi", *query, "--format=csv,noheader,nounits"],
                                   text=True, timeout=15).strip().splitlines()


def inspect_base(model, torch, layout):
    # Hash the actual resident base *after* any attachment, excluding only LoRA factors.
    params = [(name.replace(".base_layer.", "."), p) for name, p in model.named_parameters()
              if ".lora_" not in name]
    r.require(type(model).__name__ == "Qwen3_5ForCausalLM" and
              len({n for n,_ in params}) == len(params) and
              sum(p.numel() for _, p in params) == 26895998464, "wrong text architecture")
    inventory = []
    for name, param in sorted(params):
        r.require(param.dtype == torch.bfloat16 and param.device == torch.device("cuda", device_for(name, layout)),
                  "base dtype or placement changed")
        h = hashlib.sha256()
        for part in param.detach().view(-1).split(4 * 1024 * 1024):
            h.update(part.view(torch.uint8).cpu().numpy())
        inventory.append((name, list(param.shape), str(param.device), h.hexdigest()))
    return r.digest(inventory)


class Backend:
    def __init__(self, mode, image, uuids, expected):
        r.require(mode in MODES and r.valid_digest(image), "invalid worker role or image")
        r.require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8" and
                  os.environ.get("PYTHONHASHSEED") == "42" and os.environ.get("HF_HUB_OFFLINE") == "1",
                  "invalid process environment")
        r.require(os.environ.get("CUDA_VISIBLE_DEVICES") == ",".join(uuids), "wrong GPU allocation")
        adapted = mode in {"adapter", "disabled"}
        for path in [BASE] + ([ADAPTER] if adapted else []):
            r.require(os.statvfs(path).f_flag & os.ST_RDONLY, "model mount must be read-only")
        r.require(adapted == ADAPTER.exists(), "unexpected adapter mount")
        start = time.monotonic()
        identity = r.verify_identity(PROVENANCE, BASE, ADAPTER if adapted else None)
        r.require(not smi(["--query-compute-apps=pid"]), "GPU compute reservation busy")
        drivers = smi(["--query-gpu=uuid,driver_version"])
        r.require(sorted(line.split(",")[0].strip() for line in drivers) == sorted(uuids), "NVML device mismatch")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        r.require(torch.cuda.device_count() == 2 and torch.cuda.is_bf16_supported(), "BF16 GPU pair required")
        for i, uuid in enumerate(uuids):
            props = torch.cuda.get_device_properties(i)
            r.require(props.name == "NVIDIA L40S" and (props.major, props.minor) == (8, 9)
                      and str(props.uuid).lower().removeprefix("gpu-") == uuid[4:]
                      and torch.cuda.mem_get_info(i)[0] >= 38 * 2**30, "wrong device or insufficient VRAM")
        torch.set_num_threads(8)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, trust_remote_code=False)
        config = json.loads((BASE / "generation_config.json").read_bytes())
        self.eos = tuple(config["eos_token_id"])
        layout = placement()
        base, report = AutoModelForCausalLM.from_pretrained(
            BASE, dtype=torch.bfloat16, attn_implementation="sdpa", use_kernels=False,
            output_loading_info=True, device_map=layout, local_files_only=True, trust_remote_code=False)
        r.require(not any(v for k, v in report.items() if k != "unexpected_keys"),
                  "missing, mismatched or invalid base weights")
        discarded = sorted(report.get("unexpected_keys", []))
        r.require(all("visual." in k or k.startswith("vision_") for k in discarded),
                  "unexpected nonvision source weight")
        r.require(base.config._attn_implementation == "sdpa" and base.config.vocab_size == VOCAB
                  and not getattr(base, "peft_config", None), "invalid bare model")
        self.model = base
        if adapted:
            from peft import PeftModel, get_peft_model_state_dict
            from safetensors.torch import load_file
            self.model = PeftModel.from_pretrained(base, ADAPTER, local_files_only=True,
                                                   is_trainable=False, autocast_adapter_dtype=True)
            loaded = get_peft_model_state_dict(self.model)
            saved = load_file(str(ADAPTER / "adapter_model.safetensors"))
            r.require(len(saved) == 992 and saved.keys() == loaded.keys() and all(
                saved[k].dtype == loaded[k].dtype and torch.equal(saved[k], loaded[k].cpu()) for k in saved),
                "adapter tensors changed")
            layers = [m for m in base.modules() if hasattr(m, "lora_A")]
            r.require(len(layers) == 496 and all(not m.merged and not m.disable_adapters and
                      list(m.active_adapters) == ["default"] and
                      m.lora_A["default"].weight.device == m.lora_B["default"].weight.device == m.base_layer.weight.device
                      for m in layers), "invalid LoRA attachment")
        state = inspect_base(base, torch, layout)
        self.model.eval().requires_grad_(False)
        self.mode = mode
        self.device = self.model.get_input_embeddings().weight.device
        software = {"python": platform.python_version(), "versions": {n: importlib.metadata.version(n) for n in PACKAGES},
                    "cuda": torch.version.cuda, "drivers": sorted(drivers), "torch_build": torch.__config__.show(),
                    "optional": {n: importlib.util.find_spec(n) is not None
                                 for n in ("fla", "flash_attn", "causal_conv1d", "kernels")}}
        tokenizer = {"class": type(self.tokenizer).__name__, "template": r.sha(self.tokenizer.get_chat_template().encode()),
                     "files": identity["base_inventory_sha256"], "bos": self.tokenizer.bos_token_id,
                     "pad": self.tokenizer.pad_token_id, "runner": r.sha(Path(r.__file__).read_bytes())}
        kernel = {"profile": r.generation_profile(self.eos), "base_state": state, "discarded": discarded,
                  "backend": r.sha(Path(__file__).read_bytes()), "worker": r.sha(Path(__file__).with_name("live_worker.py").read_bytes())}
        self.runtime = r.Runtime(image, r.digest(software), r.digest(layout), r.digest(kernel),
                                 r.digest(tokenizer), self.eos, tuple(uuids))
        self.runtime.validate()
        self.common = json.loads(json.dumps(asdict(self.runtime)))
        r.require((mode == "discover" and expected is None) or self.common == expected, "runtime differs from freeze")
        self.identity = {"load_allocated_bytes": [torch.cuda.max_memory_allocated(i) for i in range(2)],
                         "load_reserved_bytes": [torch.cuda.max_memory_reserved(i) for i in range(2)],
                         "model": identity, "mode": mode, "common": self.common, "software": software,
                         "layout": layout, "kernel": kernel, "tokenizer": tokenizer, "load_s": time.monotonic()-start}

    def warmup(self):
        """Exercise the actual autocast/LoRA path on a public control before admitting jobs."""
        item = r.render_item("0"*32, "Reply with one word describing a green cube.", self.tokenizer)
        seen = set()
        def observe(module, args, result):
            seen.add(str(result.dtype))
        with ExitStack() as hooks:
            for layer in self.model.modules():
                if hasattr(layer, "lora_B"):
                    handle = layer.lora_B["default"].register_forward_hook(observe)
                    hooks.callback(handle.remove)
            result = self.decode(item.input_ids, 2)
        r.require(seen == ({"torch.bfloat16"} if self.mode == "adapter" else set()),
                  "unexpected LoRA compute path or dtype")
        return {"input_sha256": result.input_sha256, "output_sha256": r.digest(result.output_ids),
                "lora_compute": sorted(seen), "measurement": asdict(result.measurement)}

    def context(self):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        stack = ExitStack()
        stack.enter_context(self.torch.inference_mode())
        stack.enter_context(sdpa_kernel([SDPBackend.MATH]))
        stack.enter_context(self.torch.autocast("cuda", dtype=self.torch.bfloat16))
        if self.mode == "disabled":
            stack.enter_context(self.model.disable_adapter())
        return stack

    def sync(self):
        for i in range(2):
            self.torch.cuda.synchronize(i)

    def tensor(self, ids):
        return self.torch.tensor([ids], device=self.device, dtype=self.torch.long)

    def logits(self, **kwargs):
        result = self.model(**kwargs)
        r.require(tuple(result.logits.shape) == (1, kwargs["logits_to_keep"], VOCAB), "unexpected raw logit shape")
        if not self.torch.isfinite(result.logits).all().item():
            raise r.ItemFailure("nonfinite_logits")
        return result

    def decode(self, ids, limit):
        torch = self.torch
        sequence = []
        with self.context():
            torch.manual_seed(42)
            inp = self.tensor(ids)
            actual = r.digest(inp[0].tolist())
            self.sync()
            for dev in range(2):
                torch.cuda.reset_peak_memory_stats(dev)
            start = time.perf_counter()
            cache = None
            for i in range(limit):
                result = self.logits(input_ids=inp, past_key_values=cache, use_cache=True, logits_to_keep=1)
                raw = result.logits[0, -1]
                token = int(raw.argmax().item())
                if i == 0:
                    self.sync()
                    first = time.perf_counter()-start
                sequence.append(token)
                if token in self.eos:
                    break
                cache = result.past_key_values
                r.require(cache is not None, "missing decode cache")
                inp = self.tensor([token])
            self.sync()
            elapsed = time.perf_counter()-start
            measurement = r.Measurement(elapsed, first,
                tuple(torch.cuda.max_memory_allocated(i) for i in range(2)),
                tuple(torch.cuda.max_memory_reserved(i) for i in range(2)),
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        return r.Decode(actual, tuple(sequence), self.tokenizer.decode(sequence, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False), measurement)

    def option_logprobs(self, prefix, suffix):
        torch = self.torch
        with self.context():
            ids = self.tensor([*prefix, *suffix[:-1]])
            r.require(r.digest(ids[0, :len(prefix)].tolist()) == r.digest(prefix), "option input changed")
            raw = self.logits(input_ids=ids, use_cache=False, logits_to_keep=len(suffix)).logits[0].float()
            r.require(tuple(raw.shape) == (len(suffix), VOCAB), "invalid teacher-forced shape")
            return raw.log_softmax(-1).gather(1, self.tensor(suffix)[0].to(raw.device)[:, None]).flatten().tolist()

    def start_aux(self):
        self.sync()
        for dev in range(2):
            self.torch.cuda.reset_peak_memory_stats(dev)

    def aux_peaks(self):
        self.sync()
        return {"allocated_bytes": [self.torch.cuda.max_memory_allocated(i) for i in range(2)],
                "reserved_bytes": [self.torch.cuda.max_memory_reserved(i) for i in range(2)],
                "cpu_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}

    def probe(self, ids):
        # Forced neutral tokens exercise cache growth independently of normal EOS stopping.
        with self.context():
            inp = self.tensor(ids)
            step = self.logits(input_ids=inp, use_cache=True, logits_to_keep=1)
            raw = step.logits[0,-1].float()
            steps = r.NEW_TOKENS if len(ids) == r.CONTEXT-r.NEW_TOKENS else 1
            continuation = ids[-1]
            follow = self.tensor([continuation])
            for _ in range(steps):
                r.require(step.past_key_values is not None, "missing probe cache")
                step = self.logits(input_ids=follow, past_key_values=step.past_key_values,
                                   use_cache=True, logits_to_keep=1)
            terminal = step.logits[0,-1].float()
            reference_error = None
            if steps == 1:
                uncached = self.logits(input_ids=self.tensor([*ids, continuation]), use_cache=False,
                                       logits_to_keep=1).logits[0,-1].float()
                reference_error = (uncached-terminal).abs().max().item()
                r.require(uncached.argmax().item() == terminal.argmax().item() and reference_error <= 0.5, "cache reference mismatch")
            return {"input_sha256": r.digest(inp[0].tolist()), "logits_sha256": hashlib.sha256(
                raw.contiguous().cpu().numpy()).hexdigest(), "argmax": int(raw.argmax().item()),
                "terminal_logits_sha256": hashlib.sha256(terminal.contiguous().cpu().numpy()).hexdigest(),
                "checked_context_tokens": len(ids)+steps, "cache_reference_max_abs": reference_error}
