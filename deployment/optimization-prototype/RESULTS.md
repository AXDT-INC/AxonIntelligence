# Prototype results

Exhaustive validation PASS: 65,536 encodings; 36,608 valid, 256 nonfinite, 28,672 overflow. Exact V3 output and successful statistics. Independent verifier rejected 585,728 single-bit output corruptions. UBSan exhaustive tests passed. Additional regression groups are in validation-results.json.

Three trials per implementation, identical 2 MiB synthetic payloads, one logical CPU, nice 19:

| Implementation | Conversion MiB/s | Verification MiB/s | Three-pass projection |
|---|---:|---:|---:|
| V3 | 0.900 | 0.844 | 49.61 hours |
| Prototype | 87.275 | 105.223 | 26.05 minutes |

Median rates; projected speedup 114.3x. Conversion alone 9.80 minutes; each verification 8.13 minutes. Active CPU approximately 99–100% of one logical CPU. Process-local peak VmHWM 27.30 MiB, baseline 17.71 MiB; includes fixtures and both engines. ru_maxrss reported an inconsistent 282.39 MiB, likely inherited across exec, retained verbatim in evidence.

Projection covers 53,791,996,928 bytes. It excludes disk, hashing, manifests, source checks, receipts and lifecycle work. Original run projected 25.62 minutes; repeat 26.05. Short native phases (~20 ms), cache residency, synthetic distribution and production contention limit confidence. Retaining a V3 Python verification pass retains its many-hour cost.

All 93 original V3 files match initial hashes and metadata, and original inventory is restored. One draft was accidentally written in deployment root, detected and relocated to validation-draft-not-executed.txt; it is preserved as non-executable evidence, not a test entry point. No original file was changed. Production weights were not opened; no fresh model hash claim is made.

Recommendation: adopt as a candidate for independently reviewed, separately authorized integration, not as a current V3 replacement. See README.md for ABI, safety, interruption, interoperability and measurement limitations. No actual conversion is authorized or performed.
