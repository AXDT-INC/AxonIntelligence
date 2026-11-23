# Isolated CPU optimization prototype

Status: numerical candidate for independent Astra High review. Not integrated into V3; not authorized for model conversion. All data used here is synthetic. No safetensors/model reader, receipt-location replacement, staging, promotion or model publication is implemented.

## Build and reproduce

Run from this directory using the already installed GCC and Python standard library:

```sh
nice -n 19 sh build.sh
PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 -B validate.py
PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 -B benchmark.py
```

Build scratch stays in `build-tmp/`. Build replaces only the two prototype libraries. Validation and benchmark reports use exclusive creation: existing report names cause failure instead of overwrite. To repeat, first archive reports to operator-chosen versioned names inside this directory. Neither harness creates model artifacts. Both pin themselves to one allowed logical CPU, use nice 19, and pause between batches/trials. Nice is a scheduling preference, not a CPU quota; active work uses approximately one full logical CPU.

Compiler: GCC 11.4.0 on x86-64 Intel Xeon Platinum 8276. CPU offers F16C/AVX2/AVX512, but the build disables F16C and uses no half-precision hardware casts. Python 3.10.12 was used. Runtime uses the installed C/math runtime and ctypes, not Torch, NumPy or CUDA. The UBSan test library additionally requires the installed compiler sanitizer runtime. Local build hashes identify these artifacts; no cross-machine reproducible-build claim is made.

## Numerical design

`convert.c` reads explicit little-endian BF16 words and writes little-endian FP16 words. Sign/exponent/significand integer arithmetic implements round-to-nearest, ties-to-even, signed zero and gradual underflow. Every exponent-255 input and every BF16 magnitude above 65280 (the largest BF16 value below FP16's 65504 maximum) is rejected. No values are clamped or substituted.

`verify.c` does not call the converter or duplicate its rounding algorithm. It decodes the proposed half value and its two neighbors and checks whether the source lies within the nearest-value interval. Exact midpoints accept only an even result; sign bits must match, including zero. All relevant values and midpoints are exactly representable as binary64. This is algorithmically independent, but shares compiler/hardware and the statistics ABI; it is not an independent implementation of a complete artifact verifier.

Both implementations collect V3-compatible counters and errors in original element order. Sequential binary64 summation is intentionally preserved rather than parallelized or reassociated. Fast math and contraction are disabled. Non-nearest floating-point environments are rejected. Exhaustive testing compares the installed V3 struct-based converter AND its separate integer verifier. Successful statistics and output bits are exactly equal. Failure behavior is deliberately versioned: a rejected prototype block commits no statistics, whereas V3 may have accumulated the valid prefix. Do not substitute the prototype into V3 without reviewing that API difference.

`prototype.py` exposes bounded bytes operations, plus a synthetic tensor wrapper preserving name/shape and rejecting malformed lengths, shapes and dtypes. It retains no full model and writes no tensor files. Blocks are at most 64 KiB; C routines allocate only fixed-size local statistics. Python input/output copies require several block buffers (well below 1 MiB of transient payload storage), plus interpreter/library overhead. The synthetic harness intentionally retains a 2 MiB input and expected outputs for comparison. Actual streaming memory is independent of model size. Calls into C are synchronous; Python interruption can be delayed until the current bounded call returns.

Synthetic receipts use schema `axonai.synthetic-c-prototype.v1`, deterministic canonical JSON and output hashes; they are not V3 production receipts. No receipt or lifecycle policy has changed. The prototype has no crash-recovery or publication role.

## Evidence and limitations

- `validation-results.json`: exhaustive 65,536 encodings on optimized and UBSan builds, exact output/statistics comparison, 585,728 corrupt-output bit flips, multiple block sizes, scalar/multidimensional/empty tensors, malformed/truncated/trailing input, interruption propagation and deterministic receipts.
- `benchmark-results.json`: first measurement; `benchmark-results-v2.json`: repeat with `/proc/self/status` memory observations. Identical 2 MiB cyclic sequences spanning all finite nonoverflowing BF16 encodings, three trials per implementation. Each timed trial compares all output bytes and statistics. The distribution is synthetic and unusually underflow-heavy, not an inference-weight sample.
- Benchmark timings include Python/ctypes calls, C/Python numerical loops, statistics and output copies. Exclude disk, hashing, headers, source checks, manifests, receipt publication and lifecycle operations. The extrapolation is payload/(conversion rate) + twice payload/(verification rate), for 53,791,996,928 bytes. It assumes the prototype verifier replaces both numerical verification passes; retaining V3 independent verification retains its many-hour cost.
- Native trial phases are only about 20 milliseconds. Three-trial spread is measurement variability, not a statistical confidence interval. Cache residency, source distribution, sustained clocks, production contention, storage and hashing can materially increase full-model duration. No full-model throughput or memory guarantee is established.
- `ru_maxrss` may include a pre-exec inherited high-water mark; use the separately recorded Linux VmHWM/VmRSS for this process. Harness RSS is not a pure C-kernel allocation measurement.
- No external safetensors reader was installed or used. This is payload-level validation; file-format interoperability and safety remain responsibilities of the unchanged V3 implementation.
- Native ABI is for trusted callers through the validated wrapper. Arbitrary pointers/overlapping buffers/direct hostile C callers are not supported. UBSan is evidence against exercised undefined behavior, not a memory-safety proof; no ASan or fuzzing claim.
- Existing V3 source hashes, sizes, inodes and mtime/ctime are checked against the initial snapshot. Production model payloads were never opened; consequently this work does not independently rehash or revalidate the BF16 model.

Recommendation: adopt as an optimization candidate for separately authorized integration after independent review. Do not replace V3 now. Future integration must preserve its validated I/O, receipt isolation, no-overwrite promotion and lifecycle paths; add ABI/library integrity checks and rerun lifecycle and file-format tests. Keep the V3 numerical references as development oracles. Review process interruption, packaging and native-memory boundaries before authorizing any real conversion.
