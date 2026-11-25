# AxonAI-v1 release scope

## GitHub source release

AxonDAO licenses its first-party source code and documentation under Apache License, Version 2.0. AxonDAO has authorized public disclosure of the preserved twelve-commit history through `cee6fd7103302af493d6bc98bb9f681efabcf679`, including the historical run records and the bibliographic register in its draft form.

| Material | License and release scope |
| --- | --- |
| AxonDAO-owned first-party source and documentation | Covered by the root Apache-2.0 license. |
| Qwen-derived configuration, tokenizer materials, chat template and upstream documentation | Preserve the upstream Apache-2.0 terms, attribution and notices of changes. |
| LoRA adapter, BF16 and FP16 weights | Redistribution clearance is pending for the adapter and both weight exports. The source release includes selected model metadata and historical records. |
| Article attribution inventory | Disclosed as historical bibliographic metadata with known inaccuracies. Reconciliation remains pending. |
| Source articles and training datasets | Third-party materials retain their applicable Creative Commons terms, credit requirements and exceptions. Training datasets are outside the first-party license grant and await separate redistribution clearance. |

The public examples use operator-supplied settings. Historical run records retain site-specific identifiers for reproducibility. [First-party license](LICENSE) · [Third-party notices and attribution corrections](THIRD_PARTY_NOTICES.md)

## Model lineage and reproducibility

1. The completed scientific supervised fine-tuning run used **10,684 PMC-derived records** from 4,230 distinct article identifiers. The 20,000-record staging collection had a broader scope. After filtering, the project approved the PMC subset for internal training and adapter evaluation. Training completed two epochs and 668 optimizer steps. [Dataset freeze](engineering-maintenance/v1-freeze-activation-v1-20281029/DATASET_FREEZE_RECEIPT.json) · [Training result](engineering-maintenance/main-sft-run-20261029T150039Z/TRAINING_RESULT.json)
2. The base was `Qwen/Qwen3.8-27B` revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The v1 LoRA adapter was merged using FP32 arithmetic and rounded to BF16. The text-only export uses `Qwen3_5ForCausalLM` and contains 26,895,998,464 parameters in 851 tensors across 14 Safetensors shards. [Merge provenance](models/merged/AxonAI-v1-27B/PROVENANCE.json)
3. The FP16 export was converted from the verified BF16 checkpoint. An independent verifier compared every converted element against its expected binary16 representation. [Conversion verification](models/merged/AxonAI-v1-27B-FP16/.conversion/verification.json)

The CPU record verifier checks hashes and count relationships across the tracked evidence. An independent assembly replay reproduced the staging outputs from pinned intermediates. Repeating training requires the private inputs, pinned base and matching runtime. A complete independent replay from raw sources through training remains unverified. [Reproducibility guide](deployment/reproducibility/README.md)

## Validated capabilities and limits

| Export | Tested configuration and observation |
| --- | --- |
| BF16 | Two 48 GB L40S GPUs, up to 4,096 combined input and output tokens in the validated example. Of six comparison prompts, three merged outputs matched the base-plus-adapter outputs exactly. Top-1 predictions matched at 191 of 192 aligned positions. [BF16 validation](models/merged/AxonAI-v1-27B/VALIDATION_SUMMARY.json) |
| FP16 | Two 32 GB V100 GPUs, resident weights, batch one, sequential independent requests, up to 2,048 rendered input tokens and 256 generated tokens. The production service returned three correct responses through its private Unix socket, including a 255-token response in 37.155 seconds. Readiness returned HTTP 200 after each response. [Production verification](deployment/v100-inference/README.md#production-verification-on-2026-11-25) |

These checks establish numerical and serving behavior in the tested configurations. The project has not demonstrated scientific superiority over the base model or established clinical validity. Scientific answers and citations require independent verification. The exports support text input and output.

## Future model release

The merged weights require a separate licensing and publication decision covering the adapter and both derivatives. The existing rights record documents authorization for internal use. Before any weight release, reconcile the attribution inventory, approve the derivative terms and release manifest, and validate each precision's public loading instructions.

The separate FP16 materials describe the tested V100 configuration. The original FP16 package inherited a BF16 card, notice, dependency record and BF16-loading example. Build any eventual public package in a separate staging directory and generate its own manifest. The production loader checks a pinned manifest digest. [FP16 public card draft](deployment/public-release/fp16/README.md) · [FP16 modification notice](deployment/public-release/fp16/NOTICE.txt) · [Production manifest check](deployment/v100-inference/placement.py)
