---
library_name: transformers
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
tags:
  - biomedical
  - lora
  - transformers
---

# AxonAI-v1-27B

AxonAI-v1 is an experimental, text-only, scientifically fine-tuned language model. BF16 is the canonical release candidate. It contains 26,895,998,464 parameters in 14 Safetensors shards. Distribution of these weights awaits release-specific approval.

## Training and lineage

AxonDAO fine-tuned a LoRA adapter on 10,684 single-answer examples derived from 4,230 PubMed Central articles. Supervised loss covered assistant answer tokens and the terminal token. The prompt and empty thinking prefix were masked. The run used BF16 FSDP on eight devices, rank 16, alpha 32 and dropout 0.05. Two epochs took 668 optimizer steps with effective batch size 32, AdamW, learning rate 0.00002, 21 warmup steps and cosine decay. The frozen run had no held-out validation split. [Provenance](PROVENANCE.json) · [Article attribution candidate](SOURCE_ATTRIBUTIONS.rc.json)

The base is Qwen/Qwen3.8-27B at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The merge applied LoRA updates to 496 text weights in FP32 and rounded the result to BF16. The 851 exported tensors implement `Qwen3_5ForCausalLM`. The vision and auxiliary MTP tensors were omitted. A local FP16 derivative was created for production V100 inference. Its conversion verifier compared all 26,895,998,464 elements with their expected binary16 representations. This repository's weight inventory covers BF16 alone. [Modification notice](NOTICE.txt)

## Inference and validation

The BF16 example ran on two L40S 48 GB GPUs with SDPA, a balanced map, a 38 GiB placement budget per device and a 4,096-token combined prompt and generation envelope. With the matching PyTorch CUDA 13.2 build and the pinned runtime installed, use the included helper from the checkpoint directory. [Runtime versions](requirements-inference.txt)

```sh
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 python standalone_inference.py \
  --model . --prompt 'What is a technical replicate?' --max-new-tokens 128
```

The merged BF16 model loaded independently of PEFT and the original base. On six comparison prompts, generated tokens matched base-plus-adapter exactly in three cases. Top predictions matched at 191 of 192 aligned positions. FP16 separately served three consecutive private API requests on two V100 32 GB GPUs with up to 2,048 formatted input and 256 output tokens, batch one. The longest response generated 255 tokens in 37.155 seconds. [Validation summary](VALIDATION_SUMMARY.json)

## Recommended system prompt

Use the following text as the first message with role `system` when applying the tokenizer's chat template. This is the authoritative prompt recommendation for this release. The historical inference example above uses the shorter prompt from validation.

```text
You are AxonAI-v1, AxonDAO's experimental, text-only 27B scientific research assistant, fine-tuned from Qwen/Qwen3.8-27B. When asked about your identity or capabilities, give this lineage accurately. Claim only access to tools, sources and data actually available in this conversation.

Answer scientific questions in your own words. Distinguish observations, reported results, inferences and hypotheses. State material assumptions, uncertainty and limitations. Keep units and comparisons explicit when they affect the conclusion. If the available evidence cannot support a claim, say so and identify what would resolve it.

Credit the authors and source of specific findings when known. Give a citation or identifier only when you can support it. If you cannot verify a reference, say so. Never invent a citation, quotation, result or attribution.

When asked to recall, continue or reconstruct a copyrighted passage from an article or training example, offer an original summary of its scientific ideas and a source reference when available. Avoid verbatim reproduction, close paraphrase and piecemeal reconstruction across turns. Keep private training examples and underlying research records private.
```

## Intended use and limitations

Use this model for experimental text generation with independent review of scientific claims and citations. It can lose source context, invent specificity or produce incorrect reasoning. Scientific superiority over the base model and clinical validity remain unestablished.

## Rights

Qwen's Apache-2.0 license, copyright notice and original model card accompany its contribution. AxonDAO licenses its first-party code and documentation under Apache-2.0. The training dataset remains private. The corrected register contains bibliographic credits, counts and article-level permission evidence. [Upstream license](BASE_LICENSE.Apache-2.0.txt) · [Original Qwen card](BASE_MODEL_CARD.original.md) · [Modification notice](NOTICE.txt) · [First-party source license](SOURCE_LICENSE.Apache-2.0.txt) · [Corrected register](SOURCE_ATTRIBUTIONS.rc.json)

**AxonAI-v1 was fine-tuned on 10,684 curated scientific learning examples derived from 4,230 publications, rather than on complete scientific articles.** The articles served as upstream sources for example generation and verification. Neither the original articles nor the training dataset is distributed with the model. The licensing uncertainties documented below concern source-material permissions and do not constitute established copyright infringement by the released model weights.

**19 articles affecting 42 of the 10,684 training examples have unresolved article terms or license versions.**

| Article group | Articles | Examples | Open question |
| --- | ---: | ---: | --- |
| JAMA and JBRA | 15 | 25 | The article grants establish CC BY with attribution. The general license version is unspecified. |
| Wiley accepted manuscripts | 2 | 9 | The general reuse basis for these manuscript versions is unverified. One paper separately restricts access to underlying NHLS data. Its four examples use published study-level material. |
| Cell Stem Cell accepted manuscript | 1 | 4 | The manuscript grants unversioned CC BY. The publisher's CC BY 4.0 deposit applies to the version of record. |
| Discoveries, PMC13428630 | 1 | 4 | The PMC text-mining annotation points to CC BY 4.0 and the historical examples carry a CC-BY label. The article notice names Creative Commons Attribution but explicitly limits use to noncommercial purposes, as do the PDF and issue terms. The applicable standard license remains unresolved. |

The record review flagged source wording in 23 of these 42 examples, including three from Discoveries. One Cell Stem Cell answer matches an abstract sentence after text normalization. The review did not assess whether any passage can be recovered from the learned weights. The recommended prompt guides use and provides no evidence of model behavior or legal clearance. [Discoveries article](https://doi.org/10.15190/d.2026.10) · [Publisher's issue terms](https://www.discoveriesjournal.com/all-issues/2026-volume-14-issue-2-april-june)

Distribution of the BF16 weights awaits AxonDAO's release-specific authorization covering the adapter and merged contributions, applicable third-party terms, derivative licensing and the final manifest. Dataset redistribution would require its own review and authorization.
