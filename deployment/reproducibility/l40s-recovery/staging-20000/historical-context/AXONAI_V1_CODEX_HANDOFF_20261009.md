# AxonAI-v1: Codex Development Handoff (9 October 2026)

You are continuing an existing scientific LLM fine-tuning project on NVIDIA LaunchPad. Work from the actual local artifacts and preserve completed work; do not recreate the corpus or restart previous audits.

**Project root:** `/data/axonai`  
**Python environment:** `/data/axonai/.venv`  
**Target model:** Qwen3.8-27B, fine-tuned from an existing base  
**Generator:** locally available Qwen3.8-27B  
**Intermediate verifier:** locally available Qwen3-32B  
**Hardware:** 8 × NVIDIA L40S (approximately 48 GB each), Ubuntu 24.04

## Objective

Produce a scientifically rigorous, immutable, source-grounded, 20,000-example AxonAI-v1 supervised fine-tuning dataset, train Qwen3.8-27B with distributed LoRA/PEFT, independently evaluate scientific and coding performance, then publish an appropriately licensed and documented model on Hugging Face.

Scope: biomedical research; bioinformatics and computational biology; scientific computing and programming; research methodology and statistics; literature reasoning; uncertainty and evidence assessment; decentralized science and stable AxonOS technical knowledge.

## State and authoritative data

- Original 20,000-record staging baseline has SHA-256 `24889f6495d4e9e5015f7a63352fba470b9913fcb1dbeebc7d84c0eb0546b5fd` and must remain unchanged.
- A v6 WIP applies ten study-context question clarifications; v9 WIP stages nine provisional replacements. Verify exact paths and hashes from the existing manifests rather than trusting this summary.
- v10 independently re-verified four post-edit v9 candidates as preliminary passes.
- v11 generated sixteen provisionally verified replacement candidates, but independent review found duplicate clusters, task mismatch and scientific/coding issues. Intermediate verification is not final scientific acceptance.
- v12.1 attempted thirteen targeted corrections: two provisionally verified, eleven source-exhausted; the Pair 35 answer was insufficient and the Pair 39 Cutadapt command was technically incorrect and must not be accepted as written.
- v13 ranked alternative pinned source units; v13.1 exported 88 pinned excerpts across eleven pairs, followed by a source-suitability decision report.
- The latest human review provisionally found suitable source evidence for pairs 6, 8, 15, 18, 24, 28 and 41; pairs 29 and 30 lacked suitable literature-reasoning sources; pairs 31 and 38 required factual review. These decisions are **not automatic approval** to insert examples.
- Other open issues include opsin gene identifiers and questions conflating Galaxy Remote File Sources versus Storage Locations. Reconcile from existing decision ledgers instead of assuming their status.
- AxonOS is publicly available first-party open-source software at `https://github.com/AXDT-INC/AxonOS`; use exact pinned Git revisions and check file-specific licensing and attribution. Local Git checkouts are available under `/data/axonai/sources/`.
- Source diversity policy applies **per generation batch**: no more than 2 examples per source unit and 12 per document; still disclose global reuse, and preserve all other licensing, provenance, and scientific controls.

## Relevant local locations

Explore these locations, verify filenames, and build a reconciled manifest:

- `/data/axonai/datasets/axonai-v1/generated/20k-completion-production/run-v1/recovery/`
- `/data/axonai/remediation-runs/`
- `/data/axonai/verification-runs/`
- `/data/axonai/remediation-tools/`
- `/data/axonai/axonai_42_consolidated_gate_v3_20261009/`
- `/data/axonai/src/axonai_production_7200_v1.py` (original generator contract)
- `/data/axonai/models/base/`
- `/data/axonai/sources/`

Inspect stored v11, v12.1, v13 and v13.1 outputs, final review matrices, and provenance snapshots directly. Do not ask the user to repeatedly transfer materials already on the server.

## Immediate task

1. Inspect the actual on-disk project state, latest remediation decision ledgers, frozen evidence, model scripts and checkpoints.
2. Reconcile every outstanding remediation work order with exact record IDs, quotas, original source families, and verified source revisions. Distinguish provisional decisions from finalized approvals.
3. Produce one coherent, restartable remediation plan that avoids repeated heuristic ranking and blind GPU retries. Confirm scientific and task suitability of exact source excerpts first.
4. Implement reversible, CPU-tested corrections or replacements only where justified; use the generator and independent verifier selectively. Enforce cross-candidate and full-WIP duplication screening.
5. Independently check factual claims and execute/smoke-test coding examples; do not rely solely on Qwen3-32B acceptance. A stronger difficult-case quality audit is required before final SFT.
6. Construct a new versioned 20,000-record candidate preserving record count, unique questions, domain/task quotas and provenance. Complete final licensing, article attribution, semantic duplicate, scientific quality, tokenizer-length, validation-split and reproducibility audits.
7. Freeze only after every blocking requirement is resolved, then prepare a distributed LoRA/PEFT training smoke test, main training, independent evaluations and Hugging Face release.

## Non-negotiable boundaries

- Never modify or delete original production data, the protected staging baseline or existing SQLite checkpoints.
- Do not upgrade OS, kernel, drivers, CUDA, Docker or production infrastructure without explicit necessity and approval.
- Do not silently change source families, quotas, per-batch caps, scientific quality policy or licensing requirements.
- Never invent source evidence, commit revisions, licences, records or verifier approvals.
- Do not force examples when the available source material cannot support a distinct, valid training item.
- Prefer CPU-only tests; write immutable/versioned outputs, SHA-256 manifests, persistent logs and restartable checkpoints.
- Do **not** launch major GPU training, freeze the dataset or publish the model to Hugging Face without explicit user authorization.
- Before any smaller GPU generation run, show the concrete eligible work orders, source coverage, quota impact and expected costs, and obtain authorization when material.

## Collaboration and response protocol

Proceed autonomously with safe, read-only exploration and reversible testing. Consolidate operations rather than requesting a new instruction after every successful command. Present concise status, actionable scientific blockers, and important approval gates. Avoid repetitive broad audits and unnecessary file transfers.

**First action:** Read the existing state and present a consolidated minimum-work plan. Do not start GPU generation or mutate the corpus during initial exploration.
