# AxonAI-v1-27B — standalone merged research model

This directory is a self-contained **text-only BF16** export of the pinned Qwen base plus the completed biomedical AxonAI-v1 LoRA adapter. It requires neither the original base directory nor PEFT/adapter files at inference. It does not include RAG or the experimental v1.1 adapter.

The stored architecture is `Qwen3_5ForCausalLM` / `qwen3_5_text`, as used by the validated text inference path; the historical repository/path label is Qwen3.8-27B. Use the actual saved configuration. This export does not contain the multimodal vision tower and does not claim image/video support.

Fourteen Safetensors shards contain 26,895,998,464 BF16 parameters (approximately 53.79 GB of tensor data). Merge arithmetic was FP32, followed by one BF16 rounding. Separate LoRA and merged matrix operations can yield different logits or generated tokens. See VALIDATION_SUMMARY.json for measured comparisons; bitwise equivalence is not assumed.

## Load on another server

1. Transfer this complete directory, including the index, all shards, tokenizer, chat template, configuration, provenance and notices. Run the SHA-256 verifier before loading.
2. Use a separate Python environment matching DEPENDENCIES.json. Obtain compatible wheels for the exact versions in requirements-inference.txt, including the PyTorch CUDA build, from a trusted source appropriate to that server. The dependency wheels are not bundled; alternate versions/backends require revalidation.
3. The tested GPU layout is **two NVIDIA L40S 48 GB cards**, BF16, SDPA, no CPU/disk offload, 38 GiB weight-placement budget per card. Use a driver compatible with the recorded PyTorch CUDA runtime. Other GPU models, single-GPU layouts, quantization and serving engines are unvalidated.
4. Allow at least 64 GB host RAM; 128 GB is a conservative operational recommendation, not a measured minimum. Reserve 120 GB free disk if retaining an incoming copy plus the model. The creation procedure required substantially more CPU memory than inference; it need not be repeated.
5. Use the validated 4,096-token input-plus-output envelope. The example rejects oversized requests rather than silently truncating. Long-context capability implied by the upstream configuration has not been validated here.

```bash
python3 verify_manifest.py --root . --manifest SHA256_MANIFEST.json
python3 -m venv .venv
. .venv/bin/activate
# Install the pinned dependencies from the appropriate trusted wheel source.
python -m pip install -r requirements-inference.txt
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   python standalone_inference.py --model .   --prompt 'Explain why technical replicates do not increase the number of independent biological samples.'
```

The local example uses `AutoModelForCausalLM.from_pretrained(..., local_files_only=True, trust_remote_code=False)`. No PEFT package is needed for inference. No model-serving endpoint is started.

## Research limitations

This is an experimental biomedical-focused SFT model, not a clinically validated system. Prior base/adapter comparisons did not establish biomedical superiority. Missing source context, unsupported specificity, citation omissions and reasoning failures remain possible. Merging changes packaging; it does not repair knowledge, grounding or safety deficiencies. Do not treat generated statements as clinical advice or independently verified scientific conclusions.

A future retrieval service may insert evidence into user messages using this tokenizer's chat template. The model can run without retrieval, but source-specific questions should supply appropriate evidence. Retrieval indexing, freshness, permissions and citation checks remain separate services.

## Rights and status

Local internal packaging and validation were authorized. Publication, deployment and a public derivative licence were not authorized. The base licence and original notices are preserved; inherited release-rights status remains pending where previously recorded. No new rights investigation or claim of clearance was made. Keep provenance and attribution notices with any authorized copy.

The text loader exports 851 tensors. It omits 333 upstream vision tensors and 15 auxiliary MTP tensors that were also unused by the baseline AutoModelForCausalLM inference path. This is not a full multimodal or MTP-capable export.
