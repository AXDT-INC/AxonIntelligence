"""Validate the final image and real model metadata in a GPU-free runc container."""
import importlib.metadata
import json
import os
from pathlib import Path

from infer import SYSTEM
from placement import INPUT, plan, read_model
from test_loader import main as tiny_cpu_test


def main():
    assert not list(Path("/dev").glob("nvidia*"))
    assert os.environ["NVIDIA_VISIBLE_DEVICES"] == "void" and os.environ["CUDA_VISIBLE_DEVICES"] == ""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tiny_cpu_test()
    root = Path("/model")
    assert os.statvfs(root).f_flag & os.ST_RDONLY
    cfg, sizes = read_model(root)
    device_map, _ = plan(cfg, sizes, 2)
    config = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    with torch.device("meta"):
        shell = AutoModelForCausalLM.from_config(config, dtype=torch.float16, attn_implementation="sdpa")
    params = dict(shell.named_parameters())
    assert shell.__class__.__name__ == "Qwen3_5ForCausalLM" and params.keys() == sizes.keys()
    for k, v in params.items():
        assert v.is_meta and v.dtype == torch.float16 and v.numel()*2 == sizes[k], k
        assert any(k.startswith(prefix+".") for prefix in device_map), k
    tok = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    cases = json.loads(Path("/opt/inference/seven-prospective.json").read_bytes())
    counts = {}
    for case in cases:
        chat = tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":case["prompt"]}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        counts[case["id"]] = len(tok(chat, add_special_tokens=False).input_ids)
    assert len(counts) == 7 and all(0 < n <= INPUT for n in counts.values())
    assert not torch.cuda.is_initialized()
    print(json.dumps({"status":"passed", "versions":{k:importlib.metadata.version(k) for k in
          ("torch","transformers","accelerate","safetensors","numpy")}, "cuda_initialized":False,
          "compiled_architectures":torch._C._cuda_getArchFlags().split(), "meta_tensors":len(params),
          "meta_parameters":sum(v.numel() for v in params.values()), "prospective_prompt_tokens":counts}))


if __name__ == "__main__":
    main()
