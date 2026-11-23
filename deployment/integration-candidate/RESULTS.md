# Final candidate results

112 tests PASS, zero failures/errors/skips. test-results.json lists all test identities. Coverage includes all 61 baseline regressions, native fixture conversions, standalone exhaustive 65,536-encoding C/V3 comparisons, all valid encodings through IPC, supervision, substitution, provenance and lifecycle faults.

Three identical 2 MiB synthetic trials per engine:

| Measurement | Result |
|---|---:|
| Conversion with internal verification (median) | 19.76 MiB/s |
| Separate verification (median) | 37.48 MiB/s |
| Three-pass linear projection, 53,791,996,928 bytes | 66.08 minutes |
| Reference-engine projection | 50.14 hours |
| Projected speedup from measured pipeline | 45.5x |
| Worker startup range | 1.09–1.79 ms |
| Sum of process peak RSS observations | 76.62 MiB |

Kernel-only CPU seconds for 2 MiB conversion/verification: [0.022138, 0.023664]. Supervised block wall seconds: [0.023346958681941032, 0.02890574187040329]. Difference (IPC, Python wrapper/copies and scheduling): [0.001208958681941031, 0.005241741870403289] seconds. Active combined CPU roughly 91–100% of one logical CPU. RSS includes fixture generation and is an upper bound from separate process high-water marks, not simultaneous peak sampling.

Projection includes synthetic source/output hashing, headers, internal verification, fsync, synthetic atomic promotion, startup and shutdown. External receipt writes and recovery are excluded. Warm, small fixtures and synthetic values do not represent real storage pressure/distribution. Linear scaling overcounts fixed per-artifact costs and may undercount large-model I/O/contention. No production-duration bound or confidence interval is established.

Preservation PASS: all 112 pre-existing baseline files unchanged, including 93 accepted V3 files with matching metadata and all original prototype/history. audit_io is byte-identical; C numerical sources and Python numerical reference bodies unchanged; promotion and CLI ASTs unchanged. Coordinator diff records synthetic guards, engine dispatch, native provenance checks and pre-publication worker completion. All temporary fixtures/build directories are empty. No production payloads read or actual model published.

Ready for independent Astra High integration audit. Synthetic guards and worker limits remain in force. No authorization to execute actual conversion or replace V3.
