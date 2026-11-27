# Isolated FP16 benchmark worker

The worker loads one pinned 27B text checkpoint per container and accepts independent, single-turn jobs over standard input. It selects the AxonAI release or the bare Qwen text export by manifest SHA-256 and rehashes every inference shard on startup. It uses the same neutral scientific system message, tokenizer interface, FP16 loader, seed 42, greedy decoding, math SDPA, and fresh cache for both models. The input limit is 2,048 rendered tokens and the output limit is 256 new tokens. The image is pinned by its local immutable ID. [Benchmark design and checkpoint identities](../public-benchmarks.md)

The benchmark process owns both accepted manifest pins, its neutral prompt and all benchmark environment settings. It imports the ordinary placement planner and decode helpers read-only. The production loader accepts its single release manifest and exposes its original serving interface.

## Reproduce the bare Qwen export

The worker uses the pinned local image IDs and the two manifests listed in the benchmark plan. The offline preparation service builds the bare text export from the original Qwen checkpoint and the AxonAI release metadata. It checks each original file against the recorded SHA-256, converts each text tensor on CPU, independently checks the saved FP16 bit pattern, and requires the final manifest to match the worker's Qwen pin. Only the preparation output directory is writable. Raw weights and receipt logs belong on private storage.

Run from the repository root. Set the three paths to the restored Qwen source, verified AxonAI release, and a private output directory writable by UID 1000. On this host the original Qwen files are already restored. On a fresh host, retrieve the 27 base files in the [provenance inventory](../../../models/merged/AxonAI-v1-27B/PROVENANCE.json) from the [pinned upstream revision](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0). The original adapter is outside this inventory. The export reads only the recorded files. Use the pinned [CPU image recipe](../../fp16-docker/Dockerfile) and [inference image recipe](../../v100-inference/Dockerfile) with their adjacent wheel locks when provisioning another host. Record and independently qualify any rebuilt image with a different ID.

```bash
set -eu
umask 077
export BENCH_SOURCE=/data/axonai/receipts/qwen-benchmark/Qwen3.8-27B
export BENCH_LAYOUT=/data/axonai/models/merged/AxonAI-v1-27B-FP16
export BENCH_OUTPUT=/data/axonai/receipts/reproduce-qwen
mkdir -p "$BENCH_OUTPUT"
file=deployment/scientific-evaluation/fp16-worker/compose.prepare.yaml
docker compose -f "$file" run --rm --no-deps -T prepare fixture
docker compose -f "$file" run --rm --no-deps -T prepare export
```

## CPU qualification

Run from the repository root with Docker Compose. Each invocation mounts one checkpoint read-only and exits after synthetic qualification. Both calls use four CPU equivalents, low CPU and I/O priority, a 72 GiB memory limit, no swap, a read-only root, no privileges, and no network, GPU devices or Docker socket. The only persistent results are the JSON Lines the caller captures. Retain stdout and the exit status. Docker and library diagnostics go to stderr.

```bash
set -eu
worker=deployment/scientific-evaluation/fp16-worker
ax_model=/data/axonai/models/merged/AxonAI-v1-27B-FP16
qw_model=/data/axonai/receipts/qwen-benchmark/export-work/Qwen3.8-27B-text-FP16
BENCH_MODEL="$qw_model" docker compose -f "$worker/compose.yaml" run --rm --no-deps -T worker qwen qualify
BENCH_MODEL="$ax_model" docker compose -f "$worker/compose.yaml" run --rm --no-deps -T worker axonai qualify
```

CPU qualification loads and checks every named parameter on CPU. It runs a synthetic four-token request twice through the common response path. It requires exact token replay, a cached decode step and a visible answer. It also renders an exact 2,048-token fixture and verifies rejection at 2,049. CPU timings describe this host and execution path only.

On 27 November 2026 both FP16 checkpoints passed these checks with 851 tensors each. Their short and boundary input digests matched across arms. The host receipts are in `receipts/benchmark-worker-isolated-20261127/{axonai,qwen}.jsonl`. Qualification on a physical V100 pair remains pending.

To run the device-free regression checks, use the same Compose file, select a checkpoint, and mount the test module. Repeat with the other checkpoint. The checks cover the production-only pin, model selection, input rejection and simulated V100 admission.

```bash
BENCH_MODEL="$qw_model" docker compose -f "$worker/compose.yaml" run --rm --no-deps -T \
  --entrypoint python -v "$PWD/$worker/test_worker.py:/worker/test_worker.py:ro" \
  worker -B /worker/test_worker.py
```

## V100 qualification

Run only inside a reservation covering both physical UUIDs until the containers have exited. For a standalone Compose launch on a shared tenant host, the operator must have excluded both devices from reassignment for the entire window. A Compose device request scopes visibility. It does not communicate a reservation to AxonOS. Confirm the live allocator pool, active and credit-grace assignments, non-benchmark GPU clients and free VRAM before use. Keep production and tenant GPUs outside the reservation.

Set `BENCH_GPU_FIRST` and `BENCH_GPU_SECOND` to the reserved UUIDs in CUDA order. Use the CPU commands above with the additional Compose file `-f "$worker/compose.gpu.yaml"` and append `--device cuda` after `qualify`. Run the arms sequentially on the same pair. The worker checks both UUIDs, order, V100 architecture, free VRAM, tensor placement, finiteness and fresh-cache repeatability. The GPU qualification also forces 256 decode iterations on a private synthetic 2,048-token input, ignoring early EOS for this one stress measurement. Require exit zero, the terminal `qualified` event from each arm and matching input-ID digests across arms. Retain the per-device peaks and timings with the qualified worker version. A changed host, runtime, model manifest or GPU pair requires a new hardware receipt.

## Desktop reservations on AxonOS

A desktop lease belongs to the desktop runtime. Check the session's model mounts, supported nested isolation, available runtime and resource limits as the desktop user. A worker inside that session needs read-only weights, a pinned runtime, no network, and teardown with the lease. A standalone worker started beside a desktop can retain GPU access after the allocator ends the desktop and assigns a new tenant. An active desktop and a free-memory sample provide no lease for that worker.

For a future handover, first establish either a scheduler-managed isolated worker whose teardown is tied to the lease, or an independent allocator exclusion that covers both UUIDs through cleanup. Verify that protection before changing the desktop, and verify device ownership again before launch. The historical session-575 rollout targeted a different pair, configuration and gate image. [Historical handover](../../v100-inference/SESSION-575.md)

## Jobs and receipts

The `stream` argument opens the JSON Lines interface after model verification. Each line is one JSON object with a unique string `id`, plain user `prompt` and integer `max_new_tokens` from 1 through 256. Each `outcome` binds the arm and manifest to the item ID, status, rendered-input digest and token count. Generated answers include visible text, output IDs, stop reason, first-token and total time, and GPU memory statistics when applicable. Invalid chat delimiters and context overflow produce item-level outcomes. Invalid job schemas, duplicate IDs or runtime failures terminate the worker with a nonzero status. A caller must retain the roster and reconcile missing outcomes. The outer job scheduler owns attempt persistence and resumption. The default container timeout is 30 minutes. Size future blocks to include verification, model load and decoding. Set `BENCH_TIMEOUT` to the approved block duration when needed, leaving time for cleanup before the reservation expires.

## L40S FP16 public-suite execution

The L40S adaptation uses the same two pinned FP16 inference manifests and the same
2,048-input/256-output, batch-one, greedy, seed-42, neutral-prompt contract. The
explicit `l40s` profile requires NVIDIA L40S SM 8.9 and Torch 2.13.0+cu132 / CUDA
13.2; the original `v100` default is retained. The SM 8.6 CUDA binary is compatible
with this SM 8.9 device under the [NVIDIA Ada compatibility rules](https://docs.nvidia.com/cuda/ada-compatibility-guide/index.html#application-compatibility-on-the-nvidia-ada-gpu-architecture) and must pass the actual full-envelope qualification.

`prepare_local.py OUTPUT` derives both inference checkpoints from the original
local BF16 files without overwriting them. AxonAI conversion independently checks
every binary16 value and matches all 14 published shard hashes. Its runtime copy
contains the six inference support files and the pinned release manifest; it does
not fabricate the historical conversion timing/log files referenced by that
manifest. A separate preparation receipt records this scope. The bare Qwen export
reproduces its full pinned manifest through `prepare_base.py`.

Build `Dockerfile.l40s` with the existing venv as its context after checking the
base image ID in the recipe. Record the resulting immutable image ID. Apply
`compose.l40s.yaml` after both `compose.yaml` and `compose.gpu.yaml`; set
`BENCH_IMAGE` to that ID. No dependencies are fetched by inference containers.
The model and code mounts are read-only, the container has no network or
capabilities, and keys, datasets, answers and the host ledger are not mounted.
A kernel-created `tunl0` is accepted only when down, without addresses or routes.

The qualified local environment has Python 3.12.3, Torch 2.13.0+cu132,
Transformers 5.16.1, Accelerate 1.14.0, Safetensors 0.8.0, Tokenizers 0.23.2,
NumPy 2.5.2, PyArrow 25.0.1 and huggingface-hub 1.30.0. Host-side lexical screening
uses SciPy 1.18.1 from the private `screen-tools` directory; it is not installed
into the inference image. This recipe copies the audited existing environment;
it does not claim a portable, independently rebuilt image from a public lockfile.
Qualify any newly built image before collecting with it.

The host controller `benchmark_run.py` takes `qualify RUN_ROOT` or
`collect RUN_ROOT`. `run-config.json` binds the image, ordered GPU UUIDs, absolute
model paths, manifest hashes and profile. Qualification checks both arms on the
same pair and saves `qualification.json`; collection refuses code, input, image,
model or device-map drift. The host holds cooperative GPU locks and checks for
other GPU compute processes. These checks do not reserve a shared external
allocator; use devices assigned to this benchmark for its full lifetime.

`public_suite.py prepare RUN_ROOT --axonai MODEL --qwen MODEL` freezes the pinned
PubMedQA test, MedMCQA validation and GSM8K test splits, matching rendered input
IDs and recording context exclusions before inference. It retains source-file
hashes, original IDs, the private gold key, and a seed-42 interleaved roster.
GPQA is appended by the separate [private extension controller](../public-extensions/README.md)
after authorized access became available. `screen_public.py`
checks the available frozen SFT and staging records using the existing lexical
thresholds. The resulting incomplete-corpus flags prevent a clean-subset claim;
sealed historical cases are never read.

Four blocks use Qwen/AxonAI, AxonAI/Qwen, Qwen/AxonAI, AxonAI/Qwen order. Only one
model is resident at a time. The SQLite ledger uses full synchronous commits,
records a signed claim before dispatch, and retains every outcome. Restarting
`collect` marks unfinished claims missing and runs only untouched slots. It never
rerolls a claimed item. Logs and raw outputs stay in the private run directory.
The controller writes paired `results.json` and `completion.json` after collection;
`public_suite.py analyze RUN_ROOT` can reproduce the analysis from retained records.
The report includes eligible/returned/parsed/correct counts, operational accuracy,
paired bootstrap intervals, the primary PubMedQA exact McNemar test, missing-data
bounds, PubMedQA macro-F1/confusion, MedMCQA subject counts, latency and cap hits.
No provider APIs are involved in this two-local-arm run.

### Qualification result on this host

On 27 November 2026 both arms passed on L40S indices 1 and 2 under immutable image
`sha256:fc59b0393cfe0bd799836fd69862b8943c761da508923cf450c1472711359181`.
The Qwen 2,048-input/256-output stress request took 25.939 seconds; AxonAI took
28.392 seconds. Both peaked at 26.85 GiB reserved per device. Synthetic greedy
replay, overlength rejection, finite logits and FP32 recurrent-state checks passed.
The latest isolated CPU runs passed six worker tests and five scorer/ledger tests.

The 6,002-question / 12,004-response collection was launched after qualification.
Its private progress log is `receipts/l40s-benchmark-20261127/collection-driver.log`;
`execution.json` identifies the durable host process. Final scores must be read
from `results.json` together with `completion.json`; qualification timings are not
benchmark accuracy results. The synthetic planning estimate was 17.3 pair-hours,
including fixed and variable allowances and assumed per-suite response lengths.

GPQA Diamond is queued behind this collection on the same GPU pair. Its separate
roster has 198 planned questions, 197 eligible questions and 394 response slots;
one question exceeds the frozen 2,048-token input limit. The original roster,
ledger and frozen worker code remain unchanged. Combined aggregate results will
be written to `combined-results.json` after both phases finish. GPQA examples,
answer keys and raw records remain in the private, Git-ignored run directory.
