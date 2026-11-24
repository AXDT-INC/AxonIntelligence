# AxonAI-v1 FP16 on V100

On 2026-11-23 from 07:13:38 to 07:15:45 UTC, AxonAI-v1 FP16 ran inside AxonOS session 575 on two Tesla V100-SXM2-32GB GPUs on clu1. The recorded run exited 0, completed the initial cached decode with finite logits, and passed all seven prospective controls. Loading took 101.55 s. Sampled board memory peaked at 26,292 and 26,176 MiB. The per-case outputs, timings, fixture scores, GPU identities and model manifest digest are in the [session result](session575-smoke-20261123T0705Z/result.json).

The original Stage 1 entry point is `infer.py`. The session added checks for its private, read-only-to-the-worker model files. The first load then failed while mapping shard 2 under the worker's 72 GiB address-space limit. The successful attempt selected the safetensors `pread` backend for CPU checkpoint reads during the Transformers load. The run directory contains the two source patches in application order below, the offline wrapper and the exact session-specific launch script.

To reconstruct the successful runner for inspection, run these commands from this directory. They write a Python file in a fresh temporary directory.

```sh
d=session575-smoke-20261123T0705Z
scratch=$(mktemp -d)
cp infer.py "$scratch/infer_session.py"
patch --batch --forward --fuzz=0 "$scratch/infer_session.py" < "$d/stage-readonly.patch"
patch --batch --forward --fuzz=0 "$scratch/infer_session.py" < "$d/pread.patch"
sha256sum "$scratch/infer_session.py"
```

The reconstructed source has SHA-256 `dde53d36889beaec2cdf5138e0932c55fbb5191ec1ced51be561de41ce7539fa`. The [Stage 1 image record](stage1-20261123/result.json), the [proposed direct-run handoff](SESSION-575.md) and the planning notes below precede the successful in-session run.

## Session 584 validation on two V100s

The initial validated AxonOS Medium configuration uses batch size one, up to 2,048 input tokens and 256 generated tokens, and two V100-SXM2-32GB GPUs. The planner requires 29,892 MiB free per card at that setting. Session 584 used the staged FP16 checkpoint, offline runtime and `pread` loader. All seven regression controls passed in each of the three single-load tests on 2026-11-24. These runs establish engineering behavior for the tested prompts, including memory use, numerical stability, latency and exact answers. Scientific reasoning and domain accuracy require a separate capability evaluation.

| Input tokens | Output actual / cap | First token | Cached tokens/s | Sampled peak MiB, GPU 0 / 1 | Long-case outcome |
|---:|---:|---:|---:|---:|---|
| 1,024 | 128 / 128 | 1.55 s | 8.45 | 27,207 / 26,720 | Recalled CEDAR. The prose incorrectly described paired states as alternating at each sample. [Run record](session-584/context-1024.json) |
| 2,048 | 256 / 256 | 2.58 s | 8.69 | 28,395 / 27,908 | Correct run ID, shortest period of four, and ordered cycle. All checked logits and final FP32 recurrent states were finite. [Run record](session-584/context-2048.json) |
| 4,096 | 0 / 256 | — | — | 32,029 / 26,592 | The current implementation stopped on CUDA OOM during prefill, before producing an answer. [Run record](session-584/context-4096.json) |

### Five consecutive requests in one load

Session 584 completed five batch-one requests in 258 s, including an 84.75 s model load. Each call started with an empty cache, received its own run ID and valve pattern, and generated 256 tokens. All 15 exact checks passed across run ID, shortest period and ordered cycle. Every request had finite raw logits at each step and finite final FP32 recurrent states.

| Request | Input tokens | First token, s | Response, s | Cached tokens/s | Sampled peak MiB, GPU 0 / 1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 2.55 | 33.87 | 8.15 | 26,827 / 26,328 |
| 2 | 2,048 | 2.55 | 33.51 | 8.24 | 28,491 / 27,968 |
| 3 | 1,024 | 1.42 | 32.30 | 8.26 | 28,491 / 27,968 |
| 4 | 1,536 | 1.97 | 32.81 | 8.27 | 28,491 / 27,968 |
| 5 | 512 | 0.59 | 31.45 | 8.26 | 28,491 / 27,968 |

First-token timing starts at model prefill. Response timing includes prompt preparation and tokenization through detokenization. The first request followed model loading without a warm-up. After cache release, live PyTorch allocations rose 8.125 MiB per card on the first request and then held at 25,658.17 and 25,658.18 MiB across all five post-request measurements. Allocator reservations rose to 27,620 and 27,584 MiB on the 2,048-token request and stayed there through request five. [Run record](session-584/sustained.json)

### Reproducing a recorded test

On the Docker host, select a project-owned Medium session with the verified model and runtime already staged. Set `AXGT_SESSION_ID` and `AXGT_GPU_UUIDS` from its current allocation (two UUIDs, comma-separated in CUDA order). Export `AXGT_RUNTIME_STAGE` as the session directory containing `runtime/usr/local`. The default model path is `/home/aXonian/axonai-v100-fp16/model`. A read-only mount can be selected with `AXGT_MODEL_PATH`.

```bash
bash run-context-1024.sh "${AXGT_SESSION_ID:?}" "${AXGT_GPU_UUIDS:?}" 2048
```

The final argument selects `1024`, `2048`, `4096`, or `sustained`. Each invocation runs one test in one model load. The 4,096-token selector preserves the configuration that failed during prefill. The launcher reconstructs the worker from the pinned session 575 source and local patches. The four reviewed records include hashes of the executed worker and placement files. GPU memory arrays are in worker device order (0, 1).

## Original two-card plan

The 64-layer placement puts 25.049 GiB of weights on each device. The proposed per-device operating budget was 29.130 GiB.

## Allocation snapshot on clu1

The scheduler transaction at 2026-11-23 05:37:42 UTC and adjacent read-only NVML samples agreed with the live tenant device nodes. All eight cards were V100-SXM2-32GB, compute capability 7.0, driver 580.173.02, with 32,768 MiB reported per card. Utilization was 0% on each card and the NVML compute-process list was empty.

| Host indices | Tenant ownership | Used MiB, by index |
|---|---|---|
| 0, 1, 2, 3 | Available to the allocator. Former large session 549 ended and its container was absent. | 0, 0, 0, 0 |
| 4 | Active session 555, small | 105 |
| 5, 6 | Active session 571, medium | 150, 1 |
| 7 | Active session 572, small | 142 |

The live scheduler code matched the checkout (session_manager.py SHA-256 `9aaf2ef4df3f66b90a9ba0b8f941217745fac6a3f031100f8cc5cbb0fe5130e0`). Its configured pool is indices 0–7. The allocator serializes claims, subtracts active and eligible credit-grace allocations, and takes the first available indices. An ended row may retain its old allocation_status and GPU list. Launch requests reference sessions. No credit-grace or allocating session held 0–3 at the snapshot. Supported profile sizes are 1, 2, 4 and 8. The gate has all-card management visibility. The running tenant containers exposed exactly their assigned device nodes. Other running infrastructure comprised the gate, launcher, two Postgres instances, CAPI worker, build worker, buy bot, TLS proxy, TURN and netbird.

## Runtime

Use Docker Official Python 3.12.16 slim Bookworm on linux/amd64. Pin PyTorch 2.13.0+cu126, Transformers 5.16.1, Accelerate 1.14.0, safetensors 0.8.0, tokenizers 0.23.2, huggingface-hub 1.30.0 and NumPy 2.2.6. The local inventory contained the CPU conversion image and a CUDA 12.2 base. The dedicated GPU runtime has since passed Stage 1 CPU validation. Its immutable image ID and test logs are recorded in stage1-20261123/result.json.

The official 843,744,582-byte PyTorch wheel lists CUDA toolkit 12.6.3 and cuDNN 9.10.2.21. A CPU-only range read of its Python binding verified the member against the wheel RECORD (SHA-256 `e7dda2056403e817b572d96787adb98f12cb6742f23681af85ec1307886289f6`). Its compiled flags are `sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90`. Driver 580 supports the earlier runtime, and the cuDNN support matrix includes CUDA 12.6 and Volta. The public wheel METADATA of all 59 locked packages satisfied the Python 3.12.16 Linux/amd64 dependency graph, including CUDA extras. The approved build will check payload hashes and run pip check and a tiny CPU Transformers loading/decode test. [PyTorch release matrix](https://github.com/pytorch/pytorch/blob/v2.13.0/RELEASE.md) · [PyTorch wheel index](https://download.pytorch.org/whl/cu126/torch/) · [NVIDIA driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html) · [cuDNN 9.10.2 support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/reference/support-matrix.html)

The pinned model uses 48 gated DeltaNet and 16 full-attention layers. Its native convolution uses grouped PyTorch conv1d. Native prefill performs chunked delta updates in FP32, and cached single-token decode runs a recurrent FP32 update and returns FP16 activations. The implementation promotes the decay exponential and softplus path, and the gated norm uses FP32 arithmetic. The model keeps its original float32 SSM declaration. The text path inspected has no mandatory BF16 operation. The candidate explicitly requests FP16, math SDPA and native eager execution. Optional FlashAttention, FLA, causal-conv1d and Hub kernel packages are excluded. Triton is a declared dependency of the selected Torch wheel. No Triton kernel or torch.compile path is selected. Upstream checkpoint conversion removes the language_model prefix for this text-only architecture. The candidate checks all loaded parameter identities, sizes, FP16 dtype and device assignment. [Pinned Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py) · [Checkpoint mappings](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/conversion_mapping.py)

## VRAM plan

The planner reads safetensors headers and verifies the anchored index, model configuration and tokenizer assets. It reuses the previously verified FP16 payloads. Total tensor storage is 53,791,996,928 bytes (50.098 GiB). The embedding and separate output head are about 2.368 GiB each. Embeddings and rotary buffers start on device 0. The final norm and head share the last device. Contiguous whole-layer groups minimize the largest weight assignment. The 27 GiB cap applies to weights. GPU offload to CPU or disk is excluded.

| GPUs | Decoder layer ranges | Weights per card, GiB | Largest planned total, GiB | Margin on 32 GiB |
|---|---|---|---:|---:|
| 2 | 0–31 / 32–63 | 25.049 / 25.049 | 29.130 | 2.870 |
| 3 | 0–19 / 20–43 / 44–63 | 16.544 / 17.011 / 16.544 | 21.071 | 10.929 |
| 4 | 0–13 / 14–31 / 32–49 / 50–63 | 12.301 / 12.748 / 12.768 / 12.281 | 16.815 | 15.185 |

At batch one and 256 prompt plus 32 output tokens, the FP16 KV upper bound is 18 MiB across the model (16 layers × 288 positions × 2 K/V × 4 heads × 256 dimensions × 2 bytes). Each linear layer uses 48 × 128 × 128 FP32 recurrent elements and 10,240 × 4 FP16 convolution elements. Together those caches use 144 MiB and 3.75 MiB, respectively. The maximum per-card cache for the two-card layout is 82.875 MiB.

The plan adds 4 GiB per device for runtime context and library allocations (1.5 GiB), active-layer temporaries and workspaces (1 GiB), and loading/allocator variation (1.5 GiB). These are allowances. At 256 tokens one FP32 expanded DeltaNet Q/K/V tensor is 6 MiB, an FP32 full-attention score matrix is 6 MiB, and one FP16 MLP gate projection is 8.5 MiB. Intermediate copies and library workspaces overlap. Synchronous weight loading limits simultaneous staging. Only the last position's logits are retained by the decode loop. The runtime checks actual free VRAM against each card's budget before load. The historical BF16 smoke on two 48 GB L40S cards reached 28,483 MiB sampled on its busiest card, and the historical merged comparison reached 29,691 MiB. Different hardware, placement and kernels limit extrapolation. Three cards are a capacity comparison; this scheduler's next larger profile is four.

## CPU evidence and evaluation

- Six host unit tests passed, covering all 851 name-to-device assignments at each device count, the 27 GiB cap, cache totals, UUID/claim mismatch and busy-device rejection, and strict fixture scoring.
- The verified model manifest SHA-256 is `c3a6cb1deb5123cc35ccc9c07de36545f82b91a317f97971b6bd56b3403f2e06`. The pinned Transformers native math source tested on CPU has SHA-256 `788d4bad50a8d39be2fe79125f0f40134773cd23b1791606fb6b3ab0bc6d2263`.
- The existing CPU PyTorch image ran the native upstream function bodies offline with runc, two CPUs, 2 GiB RAM and zero GPU devices. The fixture used 48 value heads of width 128, crossed the 64-token chunk boundary, then decoded one cached token. Chunked and recurrent FP16 outputs agreed within the synthetic tolerance. FP32 states had maximum absolute difference 0.000033781 and relative L2 difference 0.00017803. The initial near-zero state tolerance was too strict for the two reduction orders. The revised test checks whole-state relative error below one FP16 epsilon and per-element error at FP16-input scale. Production finiteness checks are exact.
- Full-width grouped FP16 convolution prefill/update, causal FP16 math SDPA with 24 query and four KV heads, greedy EOS handling and rejection of NaN, positive infinity and negative infinity even in unselected logits passed on CPU.
- All seven existing prospective prompts rendered with the verified chat template and thinking disabled. An independent CPU count using the model's ByteLevel BPE rules gave 52–64 input tokens. The actual tokenizer enforces the 256-token limit at run time.
- The seven prospective cases and their scorer were copied byte-for-byte from the existing preparation workspace. Fixture SHA-256 is `c600f96c132f3b7fea0aec780776684fb2aaad7ac3f7add7dbe52772c8c41ee8`. The new tests exercise placement, reservation checks, native math and invalid logits. The BF16 validation summary retains its original JSON smoke output and six-case comparison (3/6 exact, 191/192 aligned top-1). The exact historical six prompts and reference token streams were unavailable. The published technical-replicates example remains in the original model card.
- V100 kernel execution, latency, observed VRAM, quality and historical equivalence are outstanding. Stage 1 subsequently passed the tiny hybrid HF checkpoint test on CPU, including key remapping, loading and three greedy tokens, and validated the exported image against the real tokenizer and full meta-model.

## Earlier Stage 1 build and reservation plan

1. Approve downloading the locked wheels and building the dedicated local candidate. Use a fresh tag and image ID file. No model directory is in the build context.

   ```bash
   cd /data/axonai/deployment/v100-inference
   docker buildx build --builder default --platform linux/amd64 --load \
     --iidfile image.id --tag axonai-v100-smoke:torch2.13-cu126-r1 .
   ```

2. The session 575 investigation below supersedes the earlier desktop-sharing procedure. The current desktop claim does not cover the lifecycle of a separately launched inference container. The standalone launcher has not been enabled for session 575.

3. For a future dedicated run, verify a project-owned allocation whose managed runtime contains the inference process, or an operator-managed GPU hold that outlasts both containers and prevents reassignment until cleanup. Confirm the actual assigned UUIDs, free VRAM, other allocations and the runtime-visible device list before model loading. The current launcher is specific to the earlier indices 0 and 1 and requires review for the selected procedure.

The container is nonprivileged, UID/GID 1000:1000, with all capabilities dropped, no-new-privileges, no network or Docker socket, a read-only root and a single read-only model bind. It receives only the approved UUIDs through the NVIDIA runtime. Limits are four CPU equivalents, 72 GiB RAM with zero swap, 256 PIDs, 512 MiB temporary storage, 128 MiB shared memory, and a 20-minute timeout with a 15-second kill grace. JSON lines go to stdout. Capture stdout to a new operator-owned run log. GPU allocator peaks include allocated and reserved bytes per device for load and each prompt; free VRAM samples include external usage. Generation latency includes the raw-logit finiteness checks. The first prompt is a separate four-token greedy smoke. Both a finite uncached prefill and at least one finite cached decode must complete before any of the seven scored prompts run. Exit status 2 records a quality failure after valid generation. The operator may then release their reservation through the normal scheduler.


## Session 575 desktop-sharing decision, 2026-11-23

At 06:21 UTC the live scheduler recorded project-reported session 575 as active, allocated, Medium, GPUs 1 and 2, desktop mode (SSH disabled). The scheduler container ID matched Docker. The physical UUIDs were GPU-85a39500-3cd6-4604-7ff4-4c07d48fb086 and GPU-60515032-f798-5aaa-d1e4-acf58ea8bfa8. Session 574 held GPU 0, 555 held 4, 571 held 5 and 6, and 572 held 7. GPU 3 had no active or credit-grace claimant. The lease showed 07:20:59 UTC at that snapshot.

NVML reported 32,378 and 32,494 MiB free respectively, above the 29.130 GiB per-card planning budget. Three process samples showed Xorg (host PID 974400) and xfwm4 (974993) on GPU 1. Both PIDs matched the desktop container. GPU 2 had no listed process, and neither GPU had a compute process in the sample. The desktop retains access to both physical cards. UUID scoping isolates this pair from other GPU indices but provides no independent graphics/compute or VRAM allocation within the pair.

On release or expiry the live launcher resolves and removes the single managed session container, then cleans up its network. A private-network sibling smoke container would survive that teardown and could still hold a GPU when the scheduler grants the next claim. A launch-time lease check and the smoke timeout leave this gap. A safe concurrent, zero-interference run is unsupported by the inspected deployment.

For a replacement, first arrange a supported way to supply the validated runtime and read-only model inside a scheduler-managed headless workload, or an administrative exclusion of both UUIDs through final smoke-container exit. The deployed host launcher chooses a globally configured image. Its Direct SSH mode disables graphics but does not select a per-session inference image, mount this model, or track an external sibling. After separate approval, save desktop work, end only session 575, claim the approved replacement, recheck IDs and exclusivity, and verify cleanup before returning the GPUs. Another first-fit Medium claim may receive different cards.
