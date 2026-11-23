# Reproducing AxonAI-v1 from the training corpus to V100 inference

The original L40S execution records identify both dataset writers. Starting with the pinned 20,000-record staging file, the L40S recovery reproduced the hashes of the 19,257-record candidate and the 10,684-record training JSONL in memory. The training identity binds the latter to the 668-step run. Release and session receipts connect its adapter to the merged BF16 model, the FP16 conversion and successful inference on two V100 GPUs. A separate L40S CPU reconstruction matched every byte of the 20,000-record staging file from retained intermediates. Bash history and terminal output identify the successful October 9 final assembly command, its program digest and its staging output. Both complete assembler revisions and all six original input archives now match their historical SHA-256 pins. Independent execution of the recovered assembler and full replay from raw sources remain undemonstrated.

| Stage | Verified evidence or reproducible operation | Remaining prerequisite |
|---|---|---|
| Raw sources to intermediates | The recovered 7,200-slot production command pins its program and routing manifest | Private generation inputs, historical runners, models and runtime for an independent replay |
| Intermediates to 20,000 staging records | Recorded CPU reconstruction matched all 20,000 lines and the staging SHA-256 | The comparison verifier uses 12 external files. An independent run of the recovered v1.1 assembler requires the three pinned record files, the six verified private archives and a fresh output directory. |
| Staging to frozen training data | Original inline commands, ordering, receipts and matching historical output hashes | Pinned staging file, historical hold index, licence evidence or approved manifests, and fresh isolated output paths |
| SFT and checkpointing | Verify the exact launcher, worker, settings, runtime versions and checkpoint events of the completed run | Frozen data, pinned base, eight L40S GPUs, matching environment and historical input bindings for a replay |
| Adapter merge | Apply the archived CPU merge and compare every exported tensor using the original equation checker | Complete pinned base, final adapter weights and matching packaging environment |
| BF16 verification | Verify the historical campaign through the original attempt, both repairs and sealed release summary. Verify a complete release against its pinned manifest. | A fresh comparison requires the private fixture, exact model inputs, matching runtime, two L40S GPUs and a newly authorized run window. |
| FP16 conversion | Re-run the CPU converter and independent per-element verifier on the complete BF16 release | Dedicated CPU image or its locked build inputs, an empty output and sufficient storage |
| V100 inference | Session 575 recorded finite logits and seven passing prospective controls using the FP16 manifest | Matching two-V100 runtime, FP16 payloads, positional-read loader and session inputs |

## 1. Identify the corpus before preparing examples

The recovered [inherited attribution register](../../models/merged/AxonAI-v1-27B/SOURCE_ATTRIBUTIONS.inherited-draft.json) has 4,230 distinct PMC identifiers. Its per-source counts sum to 10,684 training records. It lists 4,175 entries under CC-BY and 55 under CC0. These are the register's labels. Its release-clearance field remains open. The register records citations and counts. Reconstructing example text, ordering, labels or a split requires the original extraction.

The following original paths are relative to `/data/axonai/engineering-maintenance/` on the L40S server. Preserve the literal directory names, including the `20281029` suffixes.

| Original file | Bytes | SHA-256 |
|---|---:|---|
| `pmc-final-authorization-v1-20281029/ATTRIBUTION_REGISTER.json` | 3,118,732 | `60cea2af24eb626debe2e5815375b588eade61183c51e09d69770b7253be1761` |
| `frozen-v1-training-extraction-v1-20281029/train-10684.jsonl` | 23,478,493 | `641b43e32f3d69fa6fbf74754772e976777a26383c27de2692b90670c5dc5d07` |
| `main-sft-run-20261029T150039Z/final_adapter/adapter_config.json` | 1,255 | `f9f4066e6e969eae41b59c2273846816cbf955edfecd5fedf20fbc874ad46bc8` |
| `main-sft-run-20261029T150039Z/final_adapter/adapter_model.safetensors` | 467,047,680 | `5280cf5f30265038a6fee6aa57955b829d387b7d78fb9fe6afab86d65ac50978` |

The inherited register itself has SHA-256 `cbddcc47e0be80c8c24c44fb20ac012481a700819b217a595b7fc8007abbb340`. It records the first row above as its parent. The original rights receipt and its digest are identified in the [inherited rights record](../../models/merged/AxonAI-v1-27B/RELEASE_RIGHTS_STATUS.inherited.json). The [merge provenance](../../models/merged/AxonAI-v1-27B/PROVENANCE.json) identifies the frozen dataset and both adapter files.

The [extraction receipt](../../engineering-maintenance/frozen-v1-training-extraction-v1-20281029/EXTRACTION_RECEIPT.json) records `manifest-order record-id filter v1`, completed on October 29 at 10:28:50 UTC. The run contract pins the receipt to `c16e15e0bb1228c4f4b29ab8e0f64339c71b7a8807247dc1db3ec3d947c62b4a`. The [November 3 lineage report](../../engineering-maintenance/scope-lineage-investigation-v1-20261103/CONSOLIDATED_REPORT.md) traces these inputs and decisions:

| Stage | Original source or record | SHA-256 |
|---|---|---|
| 20,000 staging records | `datasets/axonai-v1/generated/20k-completion-production/run-v1/recovery/final-20000-staging-v11-20261009T051315Z/axonai-v1-20000-STAGING-NOT-FROZEN.jsonl` | `24889f6495d4e9e5015f7a63352fba470b9913fcb1dbeebc7d84c0eb0546b5fd` |
| 19,257 candidates after 743 exclusions | `engineering-maintenance/isolated-candidate-v1-20281029/candidate-19257.jsonl`, accompanied by `exclusion-manifest.json` | `dbaa76ba35448d6ffbc94ace43d4ad7253ac4ed415a78ff76b14abbd86ae75a1` (candidate) |
| Ordered selection of 10,684 PMC records | `engineering-maintenance/v1-revised-scope-package-v1-20281029/V1_PROPOSED_MANIFEST.json`, field `record_ids` | `b0810d15a4aa2968b03690a63a6d38d0ba9f9a9162acd357d08f77dbe772c8a4` |

### Original final assembly on October 9, 2026

At **05:13:15 UTC**, Bash history entry **1748** records the launch of `/tmp/axonai_stage_final_20000_v11.py` with three record files, six input archives and a timestamped output directory. The wrapper first verified the program’s SHA-256, `cbbb8750b0d21d129899578cfbf61ce7b1a8af665f2761277e70c55b3a9b36f9`. The terminal records the successful program check, 20,000 output lines, all five output checksums passing, and zero exit codes for both the audit and logging. Its audit JSON matches the staging audit already in Git, including the dataset SHA-256 `24889f6495d4e9e5015f7a63352fba470b9913fcb1dbeebc7d84c0eb0546b5fd`. [Exact launch command](l40s-recovery/staging-20000/original-invocation/historical-commands/history-1748.shell.txt) · [Successful terminal capture](l40s-recovery/staging-20000/original-invocation/terminal-evidence/final-v11-launch-and-success.txt)

| UTC | Bash entry | What the surviving records establish |
|---|---:|---|
| 05:08:12 | 1746 | The v1 program passed its source check. Its near-duplicate screen exceeded the unpartitioned limit at 871,180 candidate pairs and the audit exited 1. [v1 launch](l40s-recovery/staging-20000/original-invocation/historical-commands/history-1746.shell.txt) |
| 05:12:51 | 1747 | The repair required v1 SHA-256 `d62503eddd025bec1caf64b48a5e780509b0d007b8fe16446aa7ac515b8212d2`, replaced `near_duplicates`, changed the version to v1.1 and checked the resulting program digest before and after writing. The terminal records installation success. [Repair command](l40s-recovery/staging-20000/original-invocation/historical-commands/history-1747.shell.txt) · [Failure and repair capture](l40s-recovery/staging-20000/original-invocation/terminal-evidence/final-v1-failure-and-v11-repair.txt) |
| 05:13:15 | 1748 | The repaired screen completed 64 partitions, considered 871,180 candidate pairs and flagged 42. The wrapper wrote the staging audit and produced a separate four-file audit archive. [Staging audit](l40s-recovery/staging-20000/staging-audit-report.json) |

The complete command strings and timestamps came from the surviving Bash session’s in-memory history. The terminal excerpts corroborate the execution outcomes, although portions of the pasted command text are visibly truncated there. The launch **command text** hashes to `abf38137b771561941b7a122e063698ae238df51ddae409fe2169d180f17e307` as captured. The **program hash** above comes from the historical wrapper. The 2,799-byte replacement function survives intact inside the repair command and as a separate fragment. On clu1 both complete source files matched their respective launch pins. A static comparison confirmed that the recorded replacement function and version change account for every byte of v1.1. [Bash command receipts](l40s-recovery/staging-20000/original-invocation/evidence/COMMAND_RECEIPTS.json) · [Terminal capture receipt](l40s-recovery/staging-20000/original-invocation/evidence/TERMINAL_CAPTURE_RECEIPT.json) · [Repair fragment and digest](l40s-recovery/staging-20000/original-invocation/evidence/FRAGMENT_RECEIPT.json)

**All eight original artifacts have been recovered and hash-verified on clu1.** The two assembler sources are preserved byte for byte in Git. The six private archives remain in `~/axonai-recovery-review/original-assembly-inputs-20261123/` on clu1, alongside the two received source copies. The recovery receipt records each filename, original path, byte count, observed SHA-256 and historical hash authority. [v1 source](l40s-recovery/staging-20000/original-invocation/source/axonai_stage_final_20000_v1.py) · [v1.1 source](l40s-recovery/staging-20000/original-invocation/source/axonai_stage_final_20000_v11.py) · [Recovery receipt for all eight files](l40s-recovery/staging-20000/original-invocation/evidence/RECOVERED_ARTIFACTS.json)

The six complete archive digests match the original staging manifest for `recovered28_archive`, `supersession_plan`, `previous11_archive`, `single_archive`, `source12_archive` and `source1_archive`. Each archive also contains the previously verified expanded member. For example, the supersession-plan archive hashes to `b73c01129e0fb30939fca75422ba80925582ff7ce261f9334faba1919c622d5e`, and its 12-row JSONL member hashes to `495a7a415bdb7e4683ef6098ff27fa2bf39b3d867c8fc5840429f9419947b964`. The earlier L40S recovery separately verified the other three input pins for batch001, PMC and production. [Plan creation command](l40s-recovery/staging-20000/original-invocation/historical-commands/history-1721.shell.txt) · [Historical input pins](l40s-recovery/staging-20000/input-sha256.json)

The supplement `axonai-v1-20k-evidence-supplement-20261123` was checked on clu1 against manifest SHA-256 `f15e4449091610b4ee125da92ab63db870909b28720def3ef42acfb8e252b0fc`. All 41 entries matched. The preceding recovery in `b7b99d7` established byte-identical composition from retained intermediates. Its verifier and result retain their original descriptions of what was known at that time. The supplement establishes which command produced the historical output. Its target inventory records what was missing when the supplement was prepared. The subsequent recovery receipt records the eight files now verified. The October 9 terminal records the original successful run, and the later reconstruction matched its output. Independent execution of the recovered v1.1 assembler against its nine historical inputs has yet to be demonstrated. [Earlier target inventory](l40s-recovery/staging-20000/original-invocation/evidence/TARGET_ARTIFACTS.json) · [Subsequent recovery receipt](l40s-recovery/staging-20000/original-invocation/evidence/RECOVERED_ARTIFACTS.json)

**October 9 development handoff (historical context).** The project owner's incremental backup preserves a contemporaneous account of provisional corrections, rejected replacement candidates and the review still pending after staging. It cites the same 20,000-record baseline digest. Its directives and model plans applied to that October 9 session only. The later freeze, training and release receipts in this guide establish what actually ran. The SHA-256 measured on the recovered 6,827-byte Markdown file is `2dce86729664c5b2d4a21b697f0ddf6d16a9bc069550376494f41283d7b8ac31`. [Archived development handoff](l40s-recovery/staging-20000/historical-context/AXONAI_V1_CODEX_HANDOFF_20261009.md)

### Reconstruction of the 20,000 staging records

On L40S, the recovery verifier compared each reconstructed line with the preserved staging JSONL and checked for trailing bytes. All **20,000 records and 41,846,228 bytes** matched SHA-256 `24889f6495d4e9e5015f7a63352fba470b9913fcb1dbeebc7d84c0eb0546b5fd`. The historical candidate writer subsequently checked that same digest before filtering. [L40S reconstruction result](l40s-recovery/staging-20000/evidence/VERIFICATION_RESULT.json) · [Candidate receipt](../../engineering-maintenance/isolated-candidate-v1-20281029/CANDIDATE_RECEIPT.json)

| Staging rows (inclusive) | Retained input in original order | Records |
|---|---|---:|
| 1–5,800 | Repaired batch001 | 5,800 |
| 5,801–12,800 | PMC batch002 | 7,000 |
| 12,801–19,960 | Accepted production records after removing 12 superseded IDs from 7,172 | 7,160 |
| 19,961–19,988 | Recoveries covering all 28 exhausted routing slots | 28 |
| 19,989–19,999 | Accepted duplicate replacements | 11 |
| 20,000 | Final replacement for slot 06926 | 1 |

The verifier preserves JSON key order, serializes with `json.dumps(row, ensure_ascii=False, separators=(',', ':'), sort_keys=False)`, appends one LF and encodes UTF-8. It also checks replacement slots, domains, tasks and source identities. The historical staging audit counted 20,000 unique IDs and 20,000 unique normalized questions. The later October 9 observation corroborates the staging metadata. The historical near-duplicate audit was outside the reconstruction. [Recovery verifier](l40s-recovery/staging-20000/tools/verify_20k_composition.py) · [October 9 observation](l40s-recovery/staging-20000/evidence/OCT09_OBSERVATION.json) · [Staging audit](l40s-recovery/staging-20000/staging-audit-report.json)

**Original upstream invocation.** Line 68 of the L40S shell history preserves the 7,200-slot production command. It pins the production source (`107f9657c4d224d7bbbdc97dc0588ab7aa867c7c38dc21ea2a737d7201a7d441`) and routing manifest (`9fa357a11754f303703a1be1f8f9c7fa190ae8fc31d7624656c228a1ea9a2097`). Its run summary records 7,172 accepted slots and 28 exhausted. The history line has no timestamp. The package also supplies the repaired batch001 assembler and PMC producer source. Their file hashes establish the recovered bytes, while their exact historical launch revisions remain unconfirmed. [Original command](l40s-recovery/staging-20000/commands/production-7200-original-shell-history.txt) · [Command receipt](l40s-recovery/staging-20000/evidence/COMMAND_RECEIPT.json) · [Run summary](l40s-recovery/staging-20000/production-run/summary.json) · [Historical source](l40s-recovery/staging-20000/historical/src/)

To repeat the byte comparison, run the following from the repository root on a host with the 12 external files in their recorded layout. Use Python 3.12.3 to match the historical recovery environment. The verifier reads those files, including the staging file used as the reference, and emits a JSON report to stdout. The original 7,200-slot command is retained as inert text. [Required paths, sizes and hashes](l40s-recovery/staging-20000/evidence/EXTERNAL_INPUTS.json)

```sh
python3.12 -B deployment/reproducibility/l40s-recovery/staging-20000/tools/verify_20k_composition.py --data-root /data/axonai
```

The incremental package was reconciled against the fourth commit, `ef3fb0f`. Its 53 entries verified against the supplied manifest SHA-256 `bae24736d3f5166261b128b8b9706813d307424c1f41395b5e4f01edfb753fc8`. The two identical staging records already in Git are reused above. On clu1 the review checked the verifier and result hashes, block counts, production bindings and checksum records. The line-by-line comparison was performed on L40S, where the private inputs reside. This establishes the identity of the staging output. Independent replay of raw-source generation and the historical final assembly remains open. [Incremental package manifest](l40s-recovery/staging-20000/package-SHA256SUMS)

### Original commands and byte-level evidence

The commands below are inline Python recovered from October 29 L40S shell execution events. Each `.sh.txt` preserves the original command bytes. Its adjoining receipt identifies the source log, line, UTC timestamp, command and raw-event hashes, exit code and output. The November 3 lineage investigation compared already-written records. The later command recovery identifies the writers themselves. [Recovered commands and receipts](l40s-recovery/recovered-commands/)

| Event line | What the historical command did |
|---|---|
| 1444, 1453 | The first attempt created the destination directory and failed an exclusion-count assertion. The successful writer intersected 743 historical holds with the staging IDs, kept the 742 present, and added one current PyMOL ID. It excluded exactly 743 records and wrote the retained source lines in order. The candidate was 40,455,509 bytes. [Writer](l40s-recovery/recovered-commands/01453-candidate-writer.sh.txt) · [Candidate receipt](../../engineering-maintenance/isolated-candidate-v1-20281029/CANDIDATE_RECEIPT.json) |
| 1592, 1804 | The first command joined candidate provenance to the PMC licence lookup. The second selected verified rows labelled CC-BY (10,527) or CC0 (157), preserving order. The L40S recovery also matched the original evidence-matrix hash. [Matrix verification](l40s-recovery/PMC_SELECTION_VERIFICATION.json) |
| 2132, 2186, 2660 | The commands wrote the proposed manifest, wrapped its ordered IDs in the freeze manifest, and recorded activation. The proposed manifest digest above binds the training contract. The wrapper’s own SHA-256 is `8cfd9297b4b047c2a56b46c5fa6953eeb667d52f335688cefe3280847dd4f550`. The freeze deferred 8,573 records. [Activation receipt](../../engineering-maintenance/v1-freeze-activation-v1-20281029/DATASET_FREEZE_RECEIPT.json) |
| 4592, 4599 | The writer resolved `record_id` with fallback to `id`, checked duplicates and coverage, and wrote compact UTF-8 JSON in manifest order with one newline per record. After closing the file it exited 1 with `UnboundLocalError` in its checksum helper. The second command hashed the completed 23,478,493-byte file and wrote its receipt, exiting 0. [Writer](l40s-recovery/recovered-commands/04592-frozen-jsonl-writer-then-hash-failure.sh.txt) · [Receipt command](l40s-recovery/recovered-commands/04599-extraction-receipt-repair.sh.txt) |

For a downstream replay, use an isolated workspace with the historical `/data/axonai` layout, read-only private inputs, and empty output locations. Start with the staging digest above and `decision-runs/axonai-v1-freeze-decisions-v1-20261012/licensing/record_decision_index_743.jsonl` (SHA-256 `245efa2005530a49a34c59727567de92a3839499e30dc879c2bb0ab9d1b854d7`). Create the candidate directory before running 1453. To reproduce the selection, supply the licence lookup pinned in the matrix verification and follow 1592 through 2186. For extraction from an existing approved selection, supply the hash-matched candidate and freeze wrapper. Run 4592 with its output directory still absent, then 4599 after confirming the recorded post-write failure. Compare the data digests above. New timestamped receipts will have different bytes.

## 2. Recover the training invocation and optimizer settings

The recorded run is `main-sft-run-20261029T150039Z`. The merge record identifies a completed v1 LoRA run with two epochs and 668 optimizer steps. Its upstream model is recorded as `Qwen/Qwen3.8-27B` at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The text loader/export architecture is `Qwen3_5ForCausalLM`. Preserve both identifiers when locating the base. The provenance enumerates the 18 upstream shards and hashes the config, tokenizer and other supporting files individually.

### Recovered implementation and settings

The recovered [v4 snapshot manifest](../../engineering-maintenance/fsdp1-main-execution-snapshot-v4-20261029/SNAPSHOT.json) has SHA-256 `5cce07110d230da8fcb84b8823459d5b7d89d3d597398395b2acbc5addb4a93d`. All eight listed files matched their SHA-256 on clu1. The manifest identifies trainer `ea099d077551ec0baade8b93db65ec3f384757806f00819d2beb165d8c8df7d6` and config `43c6d60ee6805aa1b046b0828a51efffb3ea9d8543276deafa3a1141c06d3ef8`. These exact worker and config digests appear in both the training start and successful completion bindings.

| Component | Behavior in the executed v4 snapshot |
|---|---|
| Launch | The launcher reads a hash-bound approval and contract, verifies local inputs, then supervises eight Torch distributed processes in detached tmux. It expects the original absolute paths, interpreter and host-bound base-file metadata. |
| Input and loss | One user/assistant exchange with an optional system message. The pinned chat template disables thinking. The worker checks the exact tokenized prompt prefix, masks its labels with −100, requires EOS in the supervised answer, rejects sequences over 4,096 tokens, and right-pads batches to a multiple of eight. |
| Parallelism | Eight L40S GPUs, BF16 text weights, SDPA, FSDP1 decoder-layer wrapping, activation checkpointing, seed 42 and no TF32. One example per device and four accumulation steps give a nominal global batch of 32. |
| Optimizer and schedule | Transformers Trainer with `adamw_torch`, learning rate 0.00002, weight decay 0.01, maximum gradient norm 1.0 and cosine scheduling. At 10,684 records the schedule resolves to 334 steps per epoch, 668 total and 21 warmup steps. Betas and epsilon come from the installed library defaults. |
| Checkpoints and evaluation | The config calls for a save every 250 optimizer steps; the worker also requests timed and final saves. A completed sharded checkpoint binds the worker, config, identity and train-file hashes. The run used the historical hash-bound evaluation waiver for this main-mode invocation. |

See the original [launcher](../../engineering-maintenance/fsdp1-main-execution-snapshot-v4-20261029/launcher.py), [worker](../../engineering-maintenance/fsdp1-main-execution-snapshot-v4-20261029/train.py), [configuration](../../engineering-maintenance/fsdp1-main-execution-snapshot-v4-20261029/config.fsdp1.json) and [input contract](../../engineering-maintenance/fsdp1-main-execution-snapshot-v4-20261029/contract.json). The separate transferred `main-sft-launch-package-v1-20281028` identifies itself as preparation. Its config selects FSDP2 and a 50-step save interval. The successful run identity selects the FSDP1 v4 source.

The recovered [final adapter config](../../engineering-maintenance/main-sft-run-20261029T150039Z/final_adapter/adapter_config.json) is 1,255 bytes and matches both the operator's independent SHA-256 and the BF16 provenance. It records PEFT 0.20.0, causal-LM LoRA, rank 16, alpha 32, dropout 0.05, no bias and twelve target module names. The 32/16 scale agrees with the independently verified merge factor. The [dependency manifest](../../models/merged/AxonAI-v1-27B/DEPENDENCIES.json) describes packaging. The training start records PyTorch 2.13.0+cu132, Transformers 5.16.1, Accelerate 1.14.0, PEFT 0.20.0, Datasets 5.0.1 and Safetensors 0.8.0.

### Execution identity and completion

The [launch authorization](../../engineering-maintenance/main-sft-run-20261029T150039Z/LAUNCH_AUTHORIZATION.json) names [APPROVED_MAIN_SFT.v3.json](../../engineering-maintenance/project-owner-governance-v1-20261029/APPROVED_MAIN_SFT.v3.json). Its bytes hash to `18c7ca10073e571ddca0c8a9550ba3b71d8af2edcbcc6fb8bfc0d934ee5ad6fa`, matching both the authorization and the [run identity](../../engineering-maintenance/main-sft-run-20261029T150039Z/identity.json). This **v3 approval authorizes the v4 snapshot**. It specifies the manifest path and hash, worker and config hashes, frozen training-file digest and run output path.

The L40S preflight records the resolved eight-process training command, including its config, train file and evaluation waiver. The submission records the successful detached tmux launch with the same approval SHA-256. [Final preflight](../../engineering-maintenance/project-owner-governance-v1-20261029/FINAL_PREFLIGHT.v3.json) · [Launch submission](../../engineering-maintenance/project-owner-governance-v1-20261029/LAUNCH_SUBMISSION.v3.json)

The run identity hashes to `b1c54e3d4e00511ed0d8995cb91cc3a22dbce84be97ddb37607fd0071ccb59b8`. Its code map matches the snapshot manifest. The [lifecycle](../../engineering-maintenance/main-sft-run-20261029T150039Z/lifecycle.jsonl) starts at step 0 with 10,684 train examples, zero validation examples and no resume point. Its worker, config, dataset and identity bindings match the [training result](../../engineering-maintenance/main-sft-run-20261029T150039Z/TRAINING_RESULT.json). The run completed at 01:06:02 UTC on October 30, at step 668 and epoch 2.0, with training loss 1.154888519180749.

## 3. Validate the training checkpoints and final adapter

The lifecycle contains 40 checkpoint completion events. The final event of that kind and the training result identify `checkpoints/checkpoint-668`. The worker verifies the final sharded checkpoint before switching to full-state export of `final_adapter/`. For payload verification on the original host, the worker's `verify_checkpoint` reads a checkpoint's `COMPLETE.json`, checks its bindings, file sizes and hashes, and validates the trainer step, scheduler, model/optimizer shards and per-rank RNG inventories. The recovered lifecycle proves which saves were reported. A new payload audit requires the original checkpoint files. The downstream merge verified all 992 adapter tensors.

## 4. Merge the v1 adapter into the BF16 text model

The recorded November 9 packaging sequence was:

1. Load the pinned base through its CPU BF16 text loader, retaining the original dtype per parameter. The recorded text model had 851 BF16 parameters by tensor count.
2. Promote to FP32, attach the exact FP32 v1 adapter, and apply PEFT safe merge. Restore the original parameter dtypes with one final rounding, then serialize the text-only model as sharded Safetensors.
3. For each of the 496 adapted weights, check exact equality against `BF16(FP32(loaded BF16 weight) + 2.0 * (FP32(B) @ FP32(A)))`. Check the remaining 355 tensors against their loaded base values. The historical comparison accounted for all 992 adapter tensors and all 851 outputs. The export omitted 333 vision and 15 auxiliary tensors.
4. Compare config, tensor inventory, sharding, tokenizer, chat template and output file digests with the BF16 release manifest. The packaging record names Python 3.12.3, PyTorch 2.13.0+cu132, Transformers 5.16.1, Accelerate 1.14.0, PEFT 0.20.0 and Safetensors 0.8.0. Peak RSS during the merge was 115,369,668 KiB.

The [execution freeze](../../engineering-maintenance/merged-v1-packaging-v1-20261109/EXECUTION_FREEZE.json) pins the restored [CPU merge worker](../../engineering-maintenance/merged-v1-packaging-v1-20261109/merge_cpu.py) to `af6b3c9e7171eb632c3ce703daa6fda403ea4420d9a9a4f6cea9a9958990ec03`. It uses local-only loading, CPU placement and `save_pretrained(..., safe_serialization=True, max_shard_size='4GB')` into a fresh `merged-build` directory. The [weight verifier](../../engineering-maintenance/merged-v1-packaging-v1-20261109/verify_weights.py) checks the published model directory after the campaign promotes the staged export. The [independent equation checker](../../engineering-maintenance/merged-v1-packaging-v1-20261109/verify_equations.py) uses the upstream and merged weight indexes and resolves each LoRA key against the supplied adapter. The [finalizer](../../engineering-maintenance/merged-v1-packaging-v1-20261109/finalize_package.py) binds the release metadata. The archived merge result agrees with the BF16 provenance and the recovered equation report is byte-identical to the release copy.

### Historical BF16 validation campaign

The initial freeze matches all nine original files on clu1, including the [preflight](../../engineering-maintenance/merged-v1-packaging-v1-20261109/PREFLIGHT.json), [campaign authorization](../../engineering-maintenance/merged-v1-packaging-v1-20261109/AUTHORIZATION.json) and [four-test CPU log](../../engineering-maintenance/merged-v1-packaging-v1-20261109/CPU_TESTS.log). The preflight's 29 base/adapter bindings match the successful BF16 release's provenance by path, size and SHA-256. The authorization names the same base revision and v1 adapter digest. All three freezes specify the identical 44,848-byte `VALIDATION_PROMPTS.json` (SHA-256 `932c88ba252710a40998accd1be0ef5a0d502d67743f94c814f43dd8166e5ef9`). The recovered fixture stays with the private inputs because it includes six evaluation examples and retrieved source excerpts. It also contains the fresh JSON smoke control and the decoding settings.

| Attempt | Evidence and outcome |
|---|---|
| Original | `campaign.py` ran the merge, weight verification and baseline. The standalone merged comparison encountered the original PEFT import guard. [Initial freeze](../../engineering-maintenance/merged-v1-packaging-v1-20261109/EXECUTION_FREEZE.json) |
| First repair | The archived patch transforms `validate_gpu.py` into the exact `validate_gpu.v2.py` bytes. The repaired guard passed its two focused CPU tests according to the test summary; actual model loading still needed a second repair. [Patch](../../engineering-maintenance/merged-v1-packaging-v1-20261109/VALIDATION_REPAIR.diff) · [First repair freeze](../../engineering-maintenance/merged-v1-packaging-v1-20261109/REPAIR_EXECUTION_FREEZE.json) |
| Successful repair | `resume_validation.v2.py` selected `validate_gpu.v3.py` and `no_peft_guard.v2.py`. It reused the baseline and ran the merged and smoke modes. The archived release reports three exact token sequences in six comparisons, 191/192 aligned top-1 decisions and a passing fresh smoke. [Second repair freeze](../../engineering-maintenance/merged-v1-packaging-v1-20261109/REPAIR2_EXECUTION_FREEZE.json) · [Release validation summary](../../models/merged/AxonAI-v1-27B/VALIDATION_SUMMARY.json) |

The [historical CPU summary](../../engineering-maintenance/merged-v1-packaging-v1-20261109/CPU_TEST_RESULTS.json) records nine passes. All four original logs match its sizes and SHA-256 digests. The equation report matches both the CPU summary and the sealed release. All 21 entries across the three execution freezes (9, 6 and 6, including shared files) matched the recovered bytes. The [packaging artifact manifest](../../engineering-maintenance/merged-v1-packaging-v1-20261109/ARTIFACT_MANIFEST.json), SHA-256 `69457aa564895f5a7bd59b9fd4895c7b8fe3d6d368ab6b66523aebad96cb1545`, covers all 37 supplied package members, including the private fixture. It names the same BF16 release manifest pinned in section 5 and binds the exact released validation summary (`ad5a37e614de10853b589ad8478be1adaf84ecf38cadc6e5ee8dd18716008714`). Its other entries record historical outputs, including raw responses and logits retained outside Git.

**The BF16 validation procedure and its connection to the successful release are recovered.** A fresh comparison would generate a new baseline on two L40S GPUs with the pinned base, adapter, runtime and private fixture. The original drivers have a fixed deadline of November 23, 2026 at 17:00 UTC, and both repair drivers require a preflight less than one hour old. A new run therefore requires separate authorization, an isolated destination, fresh time controls and corresponding execution freezes. Regenerating the preflight itself uses `engineering-maintenance/mixed-scope-continuation-v1-20261109/MODEL_INTEGRITY.json` and `engineering-maintenance/post-training-decision-v1-20261109/REQUEST_MATRIX.jsonl`. Resuming directly at the second repair also requires the historical baseline’s `RESULT.json`, `responses.jsonl` and `logits-0.npy` through `logits-5.npy`, plus `merged_repair1_CLEANUP.json`. The archived counts above are the original observations. Recovery checked their provenance on CPU.

To verify the campaign records without loading models, run `python3 -B deployment/reproducibility/verify_records.py`. Add `--bf16-inputs /path/to/merged-v1-packaging-v1-20261109` to check the original private prompt fixture against all three freezes as well.

## 5. Verify an existing BF16 release

The BF16 manifest SHA-256 is `d34c6389c0ca68df59e11f2a18f3616e7b335066fd8c493acf46044551e1daee`. Its 33 entries include 14 weight shards. The historical weight check recorded 26,895,998,464 parameters and 53,791,996,928 tensor bytes. The full package on clu1 includes local files excluded from Git. A fresh checkout alone cannot satisfy a full verification.

For a full, read-only payload audit, point `bf` at the complete release. The archived verifier uses `hashlib.file_digest` and Python assertions. Use Python 3.11+ with assertions enabled. This command streams approximately 54 GB.

```sh
bf=/data/axonai/models/merged/AxonAI-v1-27B
(
  cd "$bf" || exit 1
  echo 'd34c6389c0ca68df59e11f2a18f3616e7b335066fd8c493acf46044551e1daee  SHA256_MANIFEST.json' | sha256sum -c - &&
    python3.12 -IB verify_manifest.py --root . --manifest SHA256_MANIFEST.json
)
```

## 6. Convert a verified BF16 release to FP16 on CPU

The retained implementation is [deployment/fp16-docker](../fp16-docker/). Its image build pins the Python base by digest and wheel downloads by hash. It runs a synthetic conversion self-test. The entry point enforces CPU-only PyTorch 2.13.0, NumPy 2.2.6 and Safetensors 0.8.0. The converter casts BF16 tensors to FP16, rejects nonfinite results and overflow, reads back staged output, and records hashes per shard. The independent verifier compares converted words against an exhaustive BF16-to-binary16 lookup, checks names, shapes, shard indexes and metadata, and writes the final manifest.

For a future replay, use the recorded image ID from the converter workspace or build in a fresh source directory using `bash build.sh` with the locked wheels available. The [host launcher](../fp16-docker/run.sh) uses fixed clu1 model paths. On clu1 the FP16 destination already contains the successful release. To reproduce in isolation, use the launcher's Docker invocation with `source_model` set to the complete BF16 directory and `destination` set to a newly created empty directory owned by UID 1000 on a filesystem with space for a second export. Retain its read-only source mount, dedicated runtime, UID/GID, cgroup, network and CPU restrictions. On a separate host laid out with the launcher's paths and an empty destination, `bash run.sh fresh` runs conversion followed by independent verification. The `resume` mode is for a partial output from that replay.

The original FP16 [verification receipt](../../models/merged/AxonAI-v1-27B-FP16/.conversion/verification.json) reports 14 shards, 851 converted tensors and 26,895,998,464 verified elements. Compare the new shard payloads, tensor results and config with the original [FP16 manifest](../../models/merged/AxonAI-v1-27B-FP16/SHA256_MANIFEST.json). That historical manifest has SHA-256 `c3a6cb1deb5123cc35ccc9c07de36545f82b91a317f97971b6bd56b3403f2e06`. It also covers receipts containing elapsed times, so an independent run can match the tensor bytes and produce a different manifest digest. The [session 575 record](../v100-inference/README.md) identifies the subsequent two-V100 inference run and positional-read loader adjustment.

### Reconciliation with clu1

The L40S review package stopped at BF16. On clu1, the conversion receipt names the BF16 manifest in section 5, and session 575 records the exact FP16 manifest digest in section 6. The session ran with the safetensors positional-read patch, exited 0 and passed seven prospective controls with finite logits. The historical BF16 comparison used six different cases. Its three exact generated sequences and 191/192 aligned top-1 decisions are a separate result. [Conversion receipt](../../models/merged/AxonAI-v1-27B-FP16/.conversion/verification.json) · [Session 575 result](../v100-inference/session575-smoke-20261123T0705Z/result.json) · [Positional-read patch](../v100-inference/session575-smoke-20261123T0705Z/pread.patch)

The source baseline (`6220ec7`) supplies the converter and V100 evidence, the training recovery (`6e10748`) supplies the v4 training and merge identities, and the BF16 recovery (`a6d9a62`) supplies the campaign freezes and validation records. The November 23 package supplied the original dataset command events. On clu1 all 581 entries in its `SHA256SUMS` matched, and all 71 historical files at previously tracked paths matched byte for byte. The transferred manifest’s SHA-256 is `21de49c6cdff1cf964d8ee0b2184bda4d784f73427d1eb2f9d035dca65819516`.

The retained dataset verification is the `dataset` object from L40S `evidence/CPU_VERIFICATION.json` (source file SHA-256 `74ce6ea02ed3925557b76a27f46a4aa99060f1936a8222c4099fb9c696f8ba4b`). Its matching filter and serialization digests were computed on the private inputs on L40S. The clu1 review checked the package, command events, manifests and downstream receipts on CPU. Fresh generation, training and inference require the inputs and runtime listed by stage above. [L40S dataset verification](l40s-recovery/dataset-verification.json)

## Lightweight checks and recovery scope

From the repository root, these CPU-only checks require Python 3.10+; the synthetic deployment suite also imports PyYAML and packaging. The first command creates the scratch directory expected by the suite in a fresh checkout.

```sh
mkdir -p deployment/test-workspace
python3 -B deployment/reproducibility/verify_records.py
python3 -B deployment/run_tests.py
```

The new record checker hashes the two pinned manifests and the selected metadata files, recounts the recovered attribution register and reconciles tensor counts and shard indexes. It never opens weights in its default mode. With this checkout available on the L40S host, verify an original input there by adding `--archive-root /data/axonai --check-input config`. Choose `dataset`, `attribution`, `adapter` or `base` for the other recorded inputs. Each selected group streams its named files. Only results and source or configuration need to leave that host.

Recovery started at commit `6220ec7f7f9d552bbcbaddb86fdb2b8623216bd6`. The original six Codex snapshot refs point to tree `5e24c37878d43cbb94d4c6b0d5c568be1946e6d0` (8,233 paths under deployment and merged models). The reviewed baseline tree is `ac1cc76d97d8b273768010a2bbf688a6ecf70156` (233 paths). Both trees, the remaining local Git blobs and the historical deployment workspaces were examined. The audit workspaces contain conversion candidates, storage fixtures, release preparation and service experiments. The `descriptor_checkpoint` entries are supervisor test scenarios. The attribution register is a byte-for-byte recovery of snapshot blob `297ae9a0b2ce8718cd9eec4ccf63edc459506ed9`. The L40S transfers supply the original training and merge source, hash-bound launch records, lifecycle, extraction receipt and subsequent lineage analysis. The default verifier checks the training and merge bindings directly from the archived metadata.
