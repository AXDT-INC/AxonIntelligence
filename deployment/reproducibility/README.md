# Reproducing AxonAI-v1 from the training corpus to the FP16 export

The original L40S run produced a biomedical supervised fine-tuning (SFT) adapter. A later CPU merge produced the standalone BF16 text model. A separate CPU conversion produced the FP16 release used in V100 session 575. The material on clu1 supports verification of the recorded file identities and reproduction of the conversion given the complete BF16 package. The successful run’s launch approval, lifecycle and step-668 result bind all eight files in the FSDP1 v4 execution snapshot. The original BF16 merge worker and validation utilities are also archived. The remaining source gap begins upstream, at preparation of the frozen training set.

| Stage | What can be reproduced from the current evidence | Remaining prerequisite |
|---|---|---|
| Dataset preparation | Recount the attribution register and verify a supplied frozen JSONL by size and SHA-256 | Private inputs and original candidate-builder and final-extraction implementations or invocations |
| SFT and checkpointing | Verify the exact launcher, worker, settings, runtime versions and checkpoint events of the completed run | Frozen data, pinned base, eight L40S GPUs, matching environment and historical input bindings for a replay |
| Adapter merge | Apply the archived CPU merge and compare every exported tensor using the original equation checker | Complete pinned base, final adapter weights and matching packaging environment |
| BF16 verification | Verify the historical campaign through the original attempt, both repairs and sealed release summary. Verify a complete release against its pinned manifest. | A fresh comparison requires the private fixture, exact model inputs, matching runtime, two L40S GPUs and a newly authorized run window. |
| FP16 conversion | Re-run the CPU converter and independent per-element verifier on the complete BF16 release | Dedicated CPU image or its locked build inputs, an empty output and sufficient storage |

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

The receipt reports exact ID coverage and 8,573 deferred candidates. The [lineage analyzer](../../engineering-maintenance/scope-lineage-investigation-v1-20261103/analyze.py) reads the three existing corpora, checks that frozen IDs follow manifest order, compares staging and trained objects, and writes audit results in its own workspace. The historical report found every trained record JSON-object-equal to its staging record. Neither the analyzer nor the report provides a command that wrote the frozen JSONL. Object equality and order cannot establish byte-identical JSON serialization.

The analyzer identifies upstream planning, allocation and generation programs under the original `/data/axonai/src/`, including `axonai_production_7200_v1.py`, and records their hashes in the investigation’s `INPUT_MANIFEST.json`. Those programs concern earlier staging. The investigation explicitly reported finding no persisted builder for the 19,257-candidate or final 10,684-record extraction in its bounded search. With the private artifacts one can verify the approved selection and exact training-file digest. Regeneration of those bytes from raw inputs remains unverified.

**Where the writer trace ends.** The frozen receipt names a candidate digest and a manifest digest but contains no script path, source hash or shell command. The run identity, start event and approval verify the resulting training-file digest. They were created after extraction. For a targeted search on the L40S server, the analyzer names `engineering-maintenance/isolated-candidate-v1-20281029/exclusion-manifest.json` for the first transition and `engineering-maintenance/v1-freeze-activation-v1-20281029/DATASET_FREEZE_RECEIPT.json` plus `engineering-maintenance/v1-revised-scope-package-v1-20281029/V1_SCOPE_RECEIPT.json` for the ordered selection. The investigation's `scope-lineage-investigation-v1-20261103/DECISION_TRACE.json` and `INPUT_MANIFEST.json` identify contemporary records and earlier scripts. A source file that writes the bound output, or a contemporaneous command with its inputs and matching output hash, would settle attribution. References that only read or approve the files leave the writer unresolved.

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

## Lightweight checks and recovery scope

From the repository root, these CPU-only checks require Python 3.10+; the synthetic deployment suite also imports PyYAML and packaging. The first command creates the scratch directory expected by the suite in a fresh checkout.

```sh
mkdir -p deployment/test-workspace
python3 -B deployment/reproducibility/verify_records.py
python3 -B deployment/run_tests.py
```

The new record checker hashes the two pinned manifests and the selected metadata files, recounts the recovered attribution register and reconciles tensor counts and shard indexes. It never opens weights in its default mode. With this checkout available on the L40S host, verify an original input there by adding `--archive-root /data/axonai --check-input config`. Choose `dataset`, `attribution`, `adapter` or `base` for the other recorded inputs. Each selected group streams its named files. Only results and source or configuration need to leave that host.

Recovery started at commit `6220ec7f7f9d552bbcbaddb86fdb2b8623216bd6`. The original six Codex snapshot refs point to tree `5e24c37878d43cbb94d4c6b0d5c568be1946e6d0` (8,233 paths under deployment and merged models). The reviewed baseline tree is `ac1cc76d97d8b273768010a2bbf688a6ecf70156` (233 paths). Both trees, the remaining local Git blobs and the historical deployment workspaces were examined. The audit workspaces contain conversion candidates, storage fixtures, release preparation and service experiments. The `descriptor_checkpoint` entries are supervisor test scenarios. The attribution register is a byte-for-byte recovery of snapshot blob `297ae9a0b2ce8718cd9eec4ccf63edc459506ed9`. The L40S transfers supply the original training and merge source, hash-bound launch records, lifecycle, extraction receipt and subsequent lineage analysis. The default verifier checks the training and merge bindings directly from the archived metadata.
