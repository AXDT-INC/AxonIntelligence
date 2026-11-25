# Third-party notices

## Qwen base model and files

Qwen team, **Copyright 2026 Alibaba Cloud**. The base is `Qwen/Qwen3.8-27B` at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. It is licensed under Apache License, Version 2.0. Each merged-model directory retains the upstream license in `BASE_LICENSE.Apache-2.0.txt` and the upstream model card in `BASE_MODEL_CARD.original.md`. The retained license has SHA-256 `bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a`. [Pinned upstream tree](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) · [Upstream license](models/merged/AxonAI-v1-27B/BASE_LICENSE.Apache-2.0.txt) · [Original model card](models/merged/AxonAI-v1-27B/BASE_MODEL_CARD.original.md)

The chat template, tokenizer assets and model configuration originate with Qwen. AxonAI adapted the configuration and tokenizer packaging for local text-only loading, merged its scientific LoRA adapter into the text model, omitted the vision and auxiliary MTP tensors, and converted the BF16 export to FP16. Preserve the upstream license, applicable copyright and attribution notices, and notices of changes when redistributing covered material. The root Apache-2.0 license applies only to AxonDAO-owned first-party source code and documentation. [Merge provenance](models/merged/AxonAI-v1-27B/PROVENANCE.json) · [FP16 conversion receipt](models/merged/AxonAI-v1-27B-FP16/.conversion/verification.json)

## Article attribution metadata

**Historical draft with known inaccuracies.** AxonDAO has approved disclosure of the register as provenance metadata. Redistribution clearance for the underlying training dataset and model weights remains unresolved. `SOURCE_ATTRIBUTIONS.inherited-draft.json` records the source labels used during preparation of the frozen training set. Its 4,230 entries are bibliographic metadata and per-source counts. The draft supplies neither final credit lines nor permission to redistribute articles, training examples or model weights. [Historical register](models/merged/AxonAI-v1-27B/SOURCE_ATTRIBUTIONS.inherited-draft.json) · [Inherited rights record](models/merged/AxonAI-v1-27B/RELEASE_RIGHTS_STATUS.inherited.json)

For example, the register labels `PMC13463528` CC0. The publisher licenses the article text under CC BY 4.0 and applies CC0 to associated data subject to exceptions. The register also links `PMC13463330` to CC BY 4.0, while the article specifies CC BY 3.0. Three register entries have empty author lists. Check the actual material used, article-level terms, supplied credits and third-party exceptions before preparing release attributions. The dataset and model release review will address the full inventory. [Publisher's article and data terms](https://link.springer.com/article/10.1186/s13071-026-07567-y) · [PMC13463330](https://pmc.ncbi.nlm.nih.gov/articles/PMC13463330/) · [PMC copyright guidance](https://pmc.ncbi.nlm.nih.gov/about/copyright/)

## External software

The runtime depends on PyTorch, Transformers, Accelerate, Safetensors, Tokenizers, Hugging Face Hub and other packages listed in the dependency lock. Their licenses apply to their respective software. Git contains dependency declarations and hashes. Distribution of an image or wheel bundle requires reviewing the components actually included. [V100 dependency lock](deployment/v100-inference/wheels.lock)
