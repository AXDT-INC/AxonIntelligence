> V2 correction: see RECEIPT_LIFECYCLE_V2.md. External receipt paths are now
> mandatory for both production CLIs and are validated before processing.
> Historical v1 sources and evidence are preserved under audit-history/v1/.

# AxonAI-v1 audited CPU conversion preparation

Status: source package and synthetic tests only. No production conversion, model
loading, GPU access, Docker execution, image build, package installation, network
access, AxonOS change, or publication is authorized by this package.

Authoritative source: `/data/axonai/models/merged/AxonAI-v1-27B/`.
Manifest SHA-256:
`d34c6389c0ca68df59e11f2a18f3616e7b335066fd8c493acf46044551e1daee`.
The baseline restoration preflight verified every source file. Preparation only
rechecks the small manifest plus inventory/stat metadata; it does not claim a new
full weight-payload integrity scan. See `source-integrity.json` for that scope.

## Implementation and numerical contract

`convert_fp16.py` uses only the Python standard library. It never imports Torch,
NumPy, Transformers, CUDA, or the safetensors package. It reads safetensors as a
strict length-prefixed JSON header plus contiguous little-endian payload. The
production CLI fixes the source, manifest anchor and output parent; fixture tests
use internal functions with synthetic sources.

Each BF16 input is decoded exactly into binary32, then converted with CPython
`struct.pack('<e', value)` to IEEE binary16, round-to-nearest, ties-to-even.
Signed zero is preserved. Source NaN/infinity, unsupported dtype, and magnitudes
over 65,504 fail. The largest BF16 input below this limit is 65,280; the next BF16
value is 65,536 and is rejected. There is no clamping, saturation, replacement,
quantization, or model computation. Subnormal FP16 values are retained. Nonzero
inputs that round to signed zero are counted as `zeroed_nonzero`; gradual
underflow yielding a nonzero subnormal is counted as `output_subnormal`.

Per-tensor statistics record elements, changed values, zeroed values, output
subnormals, maximum absolute/relative error, sum of absolute errors, and zero
nonfinite/overflow counts for successful conversions. A nonfinite or overflow
failure never produces a PASS artifact. These counters describe weight casting,
not inference behavior. BF16 has a wider exponent range than FP16.

`verify_fp16.py` independently derives expected binary16 bits from BF16 sign,
exponent and significand using integer shifts and explicit tie handling. It does
not import the converter or use its cast function. It independently decodes
values with exponent arithmetic to recompute statistics. Shared code is limited
to filesystem/format/integrity support; configuration consistency is separately
checked. Every output element must match the reference, including signed zero.

The verifier performs complete source-manifest hash validation, output-manifest
hash validation, source/index/header/tensor reconciliation, byte-for-byte
passthrough metadata checks through hashes, numerical checks, receipt checks,
and final source/output rehashes. Source and output file identity/stat snapshots
must also remain unchanged. PASS means verified storage, metadata and specified
cast semantics; `inference_equivalence_established` is always false.

All input/output filenames are single components; duplicate JSON keys, unexpected
inventory, symlinks in any directory component or file, hardlinked files,
nonregular files, unsupported dtypes, offset gaps/overlap, truncation, trailing
payloads, shape/index mismatches and conflicting dtype metadata are rejected.
Directory descriptors pin opened paths; `O_NOFOLLOW`, `O_EXCL`, and Linux
`renameat2(RENAME_NOREPLACE)` avoid symlink following and output replacement.
Administrative ancestor directories and package source must remain trusted and
must not be maliciously renamed or modified by their owners during a run.
These tools are not a sandbox against root.

## Derived artifact and receipts

The proposed final directory is:
`/data/axonai/models/converted/AxonAI-v1-27B-fp16-v1/`.
Staging is a new mode-0700 sibling named
`.AxonAI-v1-27B-fp16-v1.staging-<random-UUID>`.
Neither directory has been created by preparation.

The derived artifact contains:

- All 14 weight shards, names/shapes/metadata preserved, dtype changed to F16.
- `model.safetensors.index.json`, byte-identical because payload widths stay 2 bytes.
- `config.json`, changing only `dtype` (and `torch_dtype` if present) to `float16`.
  `mamba_ssm_dtype=float32`, architecture, RoPE and other settings remain intact.
- Byte-identical `generation_config.json`, `tokenizer_config.json`, `tokenizer.json`,
  `chat_template.jinja`, `BASE_LICENSE.Apache-2.0.txt`, `NOTICE.txt`, and `PROVENANCE.json`.
- `SOURCE_SHA256_MANIFEST.json`: the original manifest, including lineage for files
  not copied into the derivative.
- `CONVERSION.json`: numerical statistics, cast semantics, source anchor and
  exact converter/shared-I/O/verifier source hashes.
- `OUTPUT_SHA256_MANIFEST.json`: hashes and sizes of every other artifact file;
  only this manifest excludes its own hash.

The original BF16 launch script, dependency pins and validation summaries are
not advertised as FP16 runtime artifacts. Preserved NOTICE/PROVENANCE describe
the original lineage; `CONVERSION.json` describes the derivative. Source
validation results do not transfer automatically to FP16.

Verification receipts are canonical JSON on stdout, optionally an exclusive new
file outside the artifact. They contain source/output manifest anchors and code
hashes and exclude timestamps, staging UUIDs, inode values and host-specific
paths. Identical inputs and package code produce identical verification receipts
and output files. The converter's outer success result additionally identifies
the requested artifact name. Source code changes require re-review and change
receipt identities. There is no claim of a reproducible container or dependency
lock until external artifacts are resolved and verified.

## Memory, storage and resource expectations

Payload I/O is in 128 KiB blocks, independent of tensor/model size. Header limits
are 4 MiB per shard and 16 MiB aggregate; JSON is bounded to 16 MiB per file and
model tensor count to 10,000. Tokenizer and other large passthrough files are
stream-copied, not parsed or loaded. No whole model, tensor or shard is materialized.
For this 851-tensor package, expected converter/verifier RSS is below 512 MiB;
this is an engineering budget, not a production-weight measurement. CPU Compose
services cap memory at 1 GiB and CPU at 2 cores. The measured synthetic-test peak
is in `test-results.json`; it does not establish real-model performance.

FP16 weights require the same 53,791,996,928 payload bytes (~50.10 GiB) as BF16.
The converter requires at least source-manifest payload bytes + 16 MiB metadata
allowance + 4 GiB reserve free before staging, plus free inode capacity. Plan
**110 GiB of free space** for one successful artifact plus a failed/retry staging
copy. Atomic promotion on the same filesystem does not copy the artifact.
Concurrent disk use or disk exhaustion still fails the run; staging is retained.
Host RAM/disk must be rechecked immediately before a separately authorized run.

This deliberately dependency-free reference converter has Python work per
parameter, and the independent verifier repeats numerical work. A 26.9B-parameter
run may take many hours or days. No full-model throughput benchmark exists.
Do not change the cast implementation for speed without reviewing and rerunning
the full synthetic/reference tests. Large sequential I/O can affect tenants;
use operator-approved CPU/I/O limits and scheduling before execution.

## Exact future native CPU procedure — NOT executed

CPU conversion can use the existing tested CPython 3.10.12 standard library; it
does not need the future Python 3.12 GPU runtime or any installation. Python 3.12
CPU/container execution remains a separate untested candidate. Before execution,
authorize output-directory creation and production weight reads/writes, review
`PACKAGE_SHA256.json`, recheck free storage, and keep all GPU access prohibited.

An administrator may need to create the root-owned-parent output directory for
UID/GID 1000. A fresh receipt directory under deployment must also be authorized.
No cleanup, directory creation or conversion command below ran during preparation.

```bash
# Administrator, ONLY after authorization:
sudo install -d -o cluadmin -g cluadmin -m 0755 /data/axonai/models/converted

# cluadmin, ONLY after authorization; fail rather than reuse a receipt directory:
# Operator has already approved/created /data/axonai/receipts (0700, UID/GID 1000)

# Full source hash validation runs before any staging write.
# Converter calls the separate verifier before atomic promotion.
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B /data/axonai/deployment/convert_fp16.py \
  --artifact-name AxonAI-v1-27B-fp16-v1 \
  --authorize-cpu-conversion \
  --receipt /data/axonai/receipts/conversion.json

# Separate fresh-process verification, including full source rehashing:
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B /data/axonai/deployment/verify_fp16.py \
  --artifact-name AxonAI-v1-27B-fp16-v1 \
  --receipt /data/axonai/receipts/verification.json
```

Exit 0 plus a PASS receipt is required. FAIL is machine-readable on stdout with
exit 1 (argument syntax errors use argparse's exit 2). External receipt files are
never overwritten. If receipt writing fails after promotion, the final artifact
may exist even though the command fails; independently verify it with a fresh
receipt filename instead of rerunning conversion into it. Similarly, a failure
of directory fsync after rename can leave a published artifact with uncertain
crash durability; inspect and verify before further action.

## Failure recovery and acceptance

No automatic resume or deletion is performed. Overflow/nonfinite input, disk
failure, metadata mismatch, verifier failure, or interrupt leaves partial staging;
it is not a completed artifact. Retain it for diagnosis. A separately approved
retry uses a fresh staging UUID and an unused final artifact name. Review exact
staging paths before any separately authorized removal; never use wildcard
cleanup against model directories. SIGKILL/power loss may leave only partial files.
Rollback retains the BF16 source and abandons the derivative. No symlink/current
model pointer, service, scheduler record or source file is changed by conversion.

Acceptance requires all source hashes valid before/after, exactly 851 F16 tensors
and 26,895,998,464 parameters across 14 shards, exact independent numerical casting
checks, zero nonfinite/overflow results, reviewed underflow/error counts, unchanged
configuration except declared dtype fields, identical tokenizer/template, and
fresh-process PASS verification. There is no automatic threshold asserting that
underflow is scientifically acceptable. No inference equivalence, scientific
validation or seven-case FP16 pass is established by conversion alone.

## Runtime and Compose boundary

`requirements.candidate.txt` and `runtime-policy.json` specify the requested
Python 3.12 / Torch 2.13.0+cu126 / Transformers 5.16.1 candidate. See `DEPENDENCIES.md`
and `dependency-audit.json` for every presently known gap. The source BF16 cu132
runtime cannot be reused for V100 sm_70. No wheelhouse, complete lock or immutable
image reference has been created or verified.

`compose.yaml` is an inactive project named `axonai-v1`. All five services have
profiles, `pull_policy: never`, `restart: no`, non-root UID/GID 1000, read-only
roots, no privileges/capabilities, no network and no ports. CPU conversion and
verification select `runc` explicitly because the host default is `nvidia`.
Bind mounts disable host-path creation. Output and receipts use explicit host
paths; the verifier sees model/source read-only. No AxonOS network, volume,
Docker socket, host networking, or shared IPC is referenced. API and RAG are
explicit fail-fast placeholders, not implementations. No Docker command was run.

The GPU profile contains invalid `NOT_RESERVED` UUID defaults and an invalid
authorization command argument. Starting it is not authorized. A future launch
requires separately reserved UUIDs, a verified digest-pinned image, verified model
manifest, CPU/GPU runtime audit and an expiring operator authorization JSON. A
service start itself attaches GPUs, so authorization must precede Docker creation;
the Python gate is defense in depth, not permission to start a container.

`future_inference.py` is UNEXECUTED candidate source: before ML imports it checks
operator attestations, environment UUID declarations, exact package versions and
artifact hashes. It blocks imports of Triton/FlashAttention/FLA/causal-conv1d/Hub
kernels, sets offline modes, and uses eager PyTorch, `use_kernels=False`, SDPA math,
FP16 loading and architecture-native FP32 computations. It does not call
`model.half()` or `torch.compile`. It requires exactly two sm_70 GPUs and matching
UUIDs, budgets 27 GiB weights per GPU without CPU/disk offload, and restricts the
initial smoke to batch one, 256 input tokens and 32 output tokens. Finite scores
are checked. This is not the seven frozen cases or an API.

The import block may expose a mandatory-import incompatibility in the candidate
stack; GPU UUID property representation, optional-package detection, FP16
normalization, cache behavior, and fit are also unverified. Failures must trigger
review, not automatic relaxation of policy. Architecture-internal FP32 operations
remain intentional; FP16 activations can overflow even with finite weight casts.

For a future authorization file, required fields are `authorized: true`,
`purpose: "AxonAI-v1 independent FP16 validation"`, two `exclusive_gpu_uuids`,
a future timezone-aware `expires_utc`, `scheduler_reservation_confirmed: true`,
`runtime_audit_passed: true`, `verified_output_manifest_sha256`, and
`runtime_image_digest: "sha256:<64 hex>"`. These are operator attestations, not
cryptographic grants or a live scheduler query. Host orchestration must verify
actual image digest and enforce reservations before creation.

Baseline: all eight GPUs were allocated to sessions 549 (0–3), 555 (4), 571 (5–6),
and 572 (7); this preparation does not re-query or alter them. Integration must
account for AxonOS `AXGT_GPU_DEVICE_IDS`, `axgt_sessions` active/credit-grace
reservations, scheduler advisory lock `0x41584750`, launch authorization, and
restart/reconciliation. A standalone Compose device request does not prevent
future scheduler conflicts. No GPU UUID is selected or reserved by this package.

## Re-running preparation checks

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B /data/axonai/deployment/run_tests.py
PYTHONDONTWRITEBYTECODE=1 python3 -B /data/axonai/deployment/validate_draft.py
```

Tests use only small synthetic fixtures under `deployment/test-workspace/`,
audit-block production model/GPU paths, and prohibit network/subprocess use.
Fixture directories are cleaned after testing. YAML lint uses pre-existing
PyYAML, and wheel-audit tests use pre-existing `packaging`; converter/verifier use
only the standard library. Neither official Safetensors-library interoperability,
Docker Compose schema/engine compatibility, Python 3.12 execution, nor GPU/runtime
execution is validated by these tests. Stored test and lint reports state the scope.

## Next authorization

Authorize native CPU-only execution of the reviewed package, creation of
`/data/axonai/models/converted/` and fresh staging/final output plus
`/data/axonai/receipts/`, complete production weight reads, conversion and independent
verification. Retain prohibitions on installation, network, Docker, GPUs, AxonOS
changes, and BF16 source writes. Review resulting receipts before any dependency
acquisition, image build, scheduler reservation or GPU validation authorization.
