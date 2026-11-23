"""Build-time CPU integration test for the actual CUDA-wheel image, before a GPU reservation."""


def main():
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from infer import greedy

    if list(Path("/dev").glob("nvidia*")) or "sm_70" not in torch._C._cuda_getArchFlags().split():
        raise RuntimeError("Expected a device-free build stage and Volta-capable wheel")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = Qwen3_5TextConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=32,
                            linear_num_key_heads=1, linear_num_value_heads=2,
                            linear_key_head_dim=16, linear_value_head_dim=16,
                            layer_types=["linear_attention","full_attention"], tie_word_embeddings=False)
    with TemporaryDirectory() as tmp, torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
        reference = Qwen3_5ForCausalLM(cfg).to(dtype=torch.float16).eval()
        weights = reference.state_dict()
        root = Path(tmp)
        cfg.save_pretrained(root)
        save_file({k.replace("model.","model.language_model.",1) if k.startswith("model.") else k:v
                   for k,v in weights.items()},root/"model.safetensors",metadata={"format":"pt"})
        model, report = AutoModelForCausalLM.from_pretrained(root, local_files_only=True, trust_remote_code=False,
                        dtype=torch.float16, device_map={"":"cpu"}, attn_implementation="sdpa",
                        use_kernels=False, output_loading_info=True)
        assert not any(report.values()), report
        for name,tensor in model.state_dict().items():
            assert torch.equal(tensor,weights[name]),name
        tokens,cache,_,_ = greedy(model,torch.tensor([[1,3,7,2]]),set(),limit=3)
        assert len(tokens)==3 and cache.layers[0].recurrent_states[0].dtype==torch.float32
    assert not torch.cuda.is_initialized()
    print("Passed local checkpoint key remapping and CPU HF/Accelerate prefill and decode")


if __name__ == "__main__":
    main()
