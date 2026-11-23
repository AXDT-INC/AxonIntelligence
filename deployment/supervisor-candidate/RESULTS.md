Atomic native publication update: see [ATOMIC_NATIVE_PUBLICATION.md](ATOMIC_NATIVE_PUBLICATION.md). Developer validation passed; independent acceptance required. Historical evidence follows.

Terminal-completion update: see [TERMINAL_COMPLETION.md](TERMINAL_COMPLETION.md). Current developer validation: 162 tests, 23 native scenarios, 13 crash/IPC cases. Independent acceptance required. Prior evidence follows unchanged.

Publication Gate V3 update: see [PUBLICATION_GATE_V3.md](PUBLICATION_GATE_V3.md) for the current commit/cancellation contract and validation. Earlier results below are retained historical evidence. Production conversion remains unauthorized.

# Supervisor candidate implementation results

Historical pre-gate-V2 implementation results. Current correction and acceptance evidence are in [PUBLICATION_GATE_V2.md](PUBLICATION_GATE_V2.md) and `publication-gate-v2-release/`. The original version of this report is preserved in `audit-history/pre-publication-gate-v2/RESULTS.md`.

Status: ready for independent Astra High implementation audit. This is developer validation, not independent acceptance or authorization to convert the model.

The final executable/source trust record is `trust.json`. The definitive evidence is `acceptance-evidence/`; earlier smoke-test failures and intermediate runs are retained under `development-history/`. The complete package inventory and hashes are in `INVENTORY.md` and `CANDIDATE_MANIFEST.json`.

## Architecture delivered

The childless Python bootstrap authenticates and seals both native executables, then execs the supervisor. Native C owns the coordinator, worker, pipe ends, PIDs and pidfds. Catchable Python exceptions cannot unwind those native acquisitions. START/READY/ACK/GO gate numerical work; STATUS/CANCEL and EOF support bounded cleanup. Workers do not receive model paths. Only Python performs model I/O, verification, staging, receipt handling and atomic promotion.

The native supervisor distinguishes pending launch, unacknowledged readiness, running, requested/unconfirmed termination, cleanup incomplete, and wait-confirmed reaping. Python reports UNKNOWN when it loses an authoritative reply. A sent signal is never treated as proof of reaping. Cleanup diagnostics accumulate; bounded reports explicitly disclose truncation beyond 64 details. Cleanup retries do not retry numerical operations or publication.

The Python publication/verification/filesystem modules and both C numerical routines are unchanged copies. Worker changes are limited to the pre-numerical startup gate. The post-GO seccomp function is byte-identical to the preceding worker; startup poll runs before that filter. Supervisor and coordinator are explicitly not seccomp sandboxes.

## Final tests

All **113** unit/integration tests passed, with no skips:

| Test module | Passed |
|---|---:|
| test_conversion | 34 |
| test_lifecycle | 13 |
| test_cli_cleanup | 4 |
| test_package | 10 |
| test_supervised | 48 |
| test_trust | 4 |

All **23** isolated process/protocol scenarios passed. They cover normal completion, pre/post ACK interruption, partial frames before and after START, lost READY/ACK, stopped unacknowledged worker, child creation before PID delivery, native descriptor-acquisition interruption, repeated kill/wait/close failures, unresolved cleanup followed by recovery, supervisor death, bad ABI, oversized framing, duplicate START, pre-GO numerical rejection, worker SIGKILL/SIGTERM, and extra numerical response rejection. Expected protocol/fault rejection can produce a nonzero supervisor exit while the scenario itself passes its safety assertions.

The final validation comprises 27 harness sessions: the suite, 23 scenarios, exhaustive probe, end-to-end benchmark and standalone kernel benchmark. **164 distinct processes have wait-confirmed exit evidence**, including supervisor/coordinator/worker and test-probe processes. Every harness finished with ECHILD. Created-worker/coordinator PID sets are subsets of the native/harness wait-confirmed sets. No known test PID remained present in `/proc` at the final check. The temporary fixture and build directories are empty. This count does not substitute process termination for process reaping.

The native harness uses a private child tree, subreaper adoption and its own process group; it never relies on Python `Popen.pid`, host-wide name matching or tenant process termination. Actual workers are launched/stopped/killed/reaped in the tests. Failed syscalls are simulated at deterministic native checkpoints; kernel failure completeness is not claimed.

## Numerical and publication evidence

All **65,536 BF16 encodings** match the accepted V3 reference's exact valid output bits or rejection class: 36,608 valid, 256 nonfinite rejected, and 28,672 overflow rejected. Valid numerical statistics match exactly. Signed zeros, ties-to-even, subnormals and gradual underflow are included. The separately implemented interval-based verifier accepts the valid outputs and rejects **585,728 single-bit corrupted outputs**. Every valid encoding also passes through the supervised IPC path; scalar, multidimensional and chunk-boundary fixtures pass.

The copied and native-path tests cover source immutability, receipt isolation, collisions, malformed/truncated safetensors, short writes, ENOSPC, numerical rejection, post-promotion fsync/receipt/cleanup failures, and uncertain promotion. Worker crashes and extra numerical responses prevent pre-publication promotion. Confirmed promotion survives later cleanup interruption; unknown promotion remains unknown. Recovery does not delete, overwrite or re-promote an artifact.

No independent safetensors package is installed in the inspected host interpreter. Independent safetensors/Transformers interoperability remains **UNVERIFIED**. Numerical, hashing and custom parser tests do not establish inference equivalence.

## Performance measurements

Three identical **1 MiB** synthetic trials per engine, one CPU, nice 19, warm filesystem cache. Worker startup/block microbenchmarks and a separate native kernel measurement supplement the pipeline trials.

| Measurement | V3 reference median | Native supervisor median |
|---|---:|---:|
| Conversion plus internal verification | 0.433 MiB/s | 16.98 MiB/s |
| Independent verification | 0.843 MiB/s | 32.78 MiB/s |
| Combined CPU utilization, conversion phase | 99.5% | 87.4% |
| Combined CPU utilization, verification phase | 99.5% | 98.4% |

Native throughput ranges were 16.47–17.13 MiB/s and 31.52–34.21 MiB/s respectively. Worker startup was 0.608–0.652 ms; successful worker shutdown/reaping was 0.321–0.445 ms. A complete normal bootstrap/supervisor/coordinator/worker smoke session took 87.5 ms, including startup, a numerical operation, shutdown and harness overhead; it is not a pure startup measurement.

The maximum simultaneous RSS sample was **47.13 MiB**. The conservative sum of individual process peak RSS values was **65.27 MiB**. Actual combined peak RSS was not continuously measured; those values bracket it for this run. The upper bound includes bootstrap and fixture-generation high-water marks. Production memory use is untested.

For identical 1 MiB numerical blocks, supervised conversion took 15.99 ms versus standalone C 12.86 ms; supervised verification took 15.92 ms versus 13.36 ms. Differences of 3.13 and 2.56 ms include IPC, copying, Python validation and scheduling noise; they are not isolated syscall costs. The same one-CPU/nice19 constraints apply to both measurements.

Linear scaling to the 53,791,996,928-byte payload gives **76.4 minutes** for conversion, internal verification and separate verification, versus **49.8 hours** for reference processing: approximately **39.1×**. This is a measured synthetic projection, not a production deadline. It includes synthetic hashes, file I/O, staging and fsync; excludes external receipt publication and recovery. Small fixtures include fixed per-artifact overhead, while full-model storage behavior, shard/tensor counts, cache state and contention can change the result in either direction. No host-wide cache flushing or production I/O benchmark was performed. Raw measurements and assumptions are in `performance-summary.json` and the benchmark evidence.

## Preservation and remaining limits

All **206 pre-existing deployment files** match the initial hashes and recorded stat fields, including all **93 accepted V3 files**, the standalone prototype and the complete preceding integration candidate/history. All **34 model inventory/stat entries** match the historical source record. The small source manifest still hashes to `d34c6389c0ca68df59e11f2a18f3616e7b335066fd8c493acf46044551e1daee`. No production weight payload was read; weight hashes were not recomputed.

Local repeated GCC builds were byte-identical. The trusted manifest, compiler/system libraries, Linux/CPython behavior and same-UID operator remain trust assumptions. Native executable-byte integrity is not full process isolation. No guarantee is made against SIGKILL, power loss, native corruption, unavailable scheduling or stalled kernel I/O. Outside the test subreaper, orphan reaping after supervisor loss belongs to the OS and cannot be affirmed by a dead supervisor; the coordinator reports UNKNOWN.

No GPU, CUDA, Docker, network, package installation, AxonOS or scheduler operation occurred. No actual model conversion or production publication occurred. Current synthetic size, CPU and wall-time limits deliberately disallow a production run. Next step: independent Astra High review of this candidate, followed only by separately authorized corrections or production execution planning.
