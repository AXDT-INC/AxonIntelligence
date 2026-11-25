---
library_name: transformers
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
tags:
  - biomedical
  - text-generation
---

# AxonAI-v1-27B-FP16

**Public model card draft. Weight distribution and the derivative license await separate approval.**

AxonAI-v1 is an experimental 27B language model fine-tuned on scientific instruction and response examples from biomedical literature. This text-only FP16 export contains 26,895,998,464 parameters in 851 tensors across 14 Safetensors shards.

## Origin and conversion

The base is `Qwen/Qwen3.8-27B` at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The project trained the v1 LoRA adapter on 10,684 PMC-derived examples, merged it into the base text model using FP32 arithmetic, and rounded the result to BF16. A subsequent conversion cast those tensors to IEEE FP16. The conversion verifier checked every element against its expected binary16 representation. The exported architecture is `Qwen3_5ForCausalLM` (`qwen3_5_text`). It omits the vision tower and auxiliary MTP tensors. [Conversion receipt](../../../models/merged/AxonAI-v1-27B-FP16/.conversion/verification.json)

## Tested inference

The validated service loads the FP16 checkpoint locally using Transformers 5.16.1, PyTorch 2.13.0+cu126, and `dtype=torch.float16`. It keeps the model resident across two V100 32 GB GPUs. The measured configuration uses batch size one, up to 2,048 input tokens including the chat template, and up to 256 generated tokens. The pinned runtime uses `TORCH_DISABLE_NATIVE_JIT=1` and checks the reserved GPU UUIDs at startup. Use the FP16 service instructions for this configuration. The standalone example bundled with the historical checkpoint explicitly selects BF16 and was validated for the earlier BF16 package. The inherited dependency record also describes that earlier runtime. Use the V100 wheel lock to identify the tested FP16 dependencies. [Pinned V100 wheels](../../v100-inference/wheels.lock) · [Runtime and service instructions](../../v100-inference/README.md#start-or-restore-a-private-instance)

Three consecutive production requests returned correct responses through a private Unix socket. The longest generated 255 tokens in 37.155 seconds. Readiness remained available after each request. The project has not established scientific superiority over the base model or clinical validity. Independently verify scientific claims and citations. [Production results](../../v100-inference/README.md#production-verification-on-2026-11-25)

## Rights

Qwen's Apache-2.0 terms apply to its contribution. A release decision for the adapter and merged weights is pending. The inherited article register is a historical draft with known attribution and license errors. Training dataset redistribution also awaits separate clearance. [Modification notice](NOTICE.txt) · [Third-party notices](../../../THIRD_PARTY_NOTICES.md)
