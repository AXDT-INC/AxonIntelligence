# Publication gate V2 — ready for independent review

This is a targeted correction to the cancellation-to-publication defect. It is developer validation, not independent acceptance or authorization for production conversion. Final evidence is in `publication-gate-v2-release/`; the code diff is `publication-gate-v2.diff`.

## Root cause and correction

Previously `Native.finish()` returned immediately when `closed` was true. That property proves resource cleanup, and can also be true after SIGKILL/CANCEL. The independent audit demonstrated a fully verified synthetic artifact being promoted after direct `worker.control(4)` returned a reaped worker with wait status 9.

FINISH success is now separate from resource cleanup. The proxy retains the authenticated FINISH response independently of later STATUS/CANCEL responses and validates it immediately. A subsequent clean status cannot repair an unsuccessful FINISH. An idempotent return from `finish()` requires prior validated FINISH and continued eligibility; `closed` alone never suffices.

Cancellation is sticky per worker token. Both `Native.control(CANCEL)` and direct `Session.exchange(CANCEL, token)` invalidate eligibility before attempting I/O. Abort, protocol failure and session loss also invalidate eligibility. Successful subsequent termination, reaping or descriptor cleanup cannot clear that state. The flag describes terminal cancellation/failure for publication authorization; it is not proof that a signal was delivered or a process reaped.

## Authorization invariant

Immediately before the unchanged atomic-promotion sequence, all of the following must hold:

- The same worker is bound to the same control session and the same conversion lifecycle object.
- The complete independent verification result is PASS, including source-integrity and complete-output-inventory checks.
- The lifecycle is still unpublished, in numerical verification or pre-publication, and has no recorded primary failure or FAIL status.
- FINISH was requested, its response was authenticated to the expected ABI/sequence/op/token by Session, and its PID matches the original READY worker.
- The FINISH response was validated and retained; current resource status agrees with its relevant completion fields.
- Native wait status establishes normal exit with code zero, confirmed reaping, and completed resource cleanup. Normal exit and signal termination are decoded separately with `os.WIFEXITED`, `os.WEXITSTATUS`, and `os.WIFSIGNALED`.
- No cancellation, broken/failed session, unresolved context state, pending numerical response, or native/local/session cleanup diagnostic exists.

The coordinator marks verification success only after the existing verifier returns. `finish_before_publication(result, lifecycle)` obtains authorization. A second, non-executing `require_publication_authorized(lifecycle)` check runs after staging fsync/cleanup and immediately before entering the existing atomic-promotion window. It never retries numerical operations, FINISH, or publication. Consequently, replacing the first hook with plain successful FINISH is insufficient, and cancellation during pre-publication fsync is rejected.

The existing explicit reference-engine path is retained. No automatic fallback or engine-selection change was introduced. Trusted coordinator code remains a prerequisite; these checks are not a sandbox against arbitrary Python code replacing both gates or the publication routine.

## Independent lifecycle facts

| Event | Recorded facts and behavior |
|---|---|
| Numerical DATA | Work is incomplete for publication; no numerical work is accepted after a FINISH request. |
| Complete verifier PASS | `numerical_work_completed` becomes true for the bound conversion lifecycle. |
| FINISH sent | `finish_requested` becomes true; no retransmission after an uncertain result. |
| Authenticated FINISH response | `finish_acknowledged` becomes true; a copy is retained as `finish_response`. |
| All FINISH prerequisites pass | `finish_validated` becomes true; repeated successful `finish()` sends no new frame. |
| CANCEL, abort, protocol/session failure | Eligibility is terminally invalidated. Cleanup may continue, but cannot restore authorization. |
| Wait/cleanup success | `reaped` and `cleanup_complete` remain resource facts, independent of job success. |
| Nonzero exit or signal | Separately decoded failure facts; new publication denied. |
| Missing/uncertain status | `worker_exit_unknown` is true; new publication denied. |
| Promotion confirmed by V3 | The original shared lifecycle's `promoted=True` remains authoritative, separately from durability. |
| Promotion outcome unknown | The original `promoted=None` and durability state remain unknown. |
| Later cleanup/cancellation | Authorization for new publication is false; already confirmed or unknown publication facts are preserved. |

The proxy observes the original publication lifecycle by reference; it never writes or clears its promotion/durability facts. No recovery path deletes, overwrites or re-promotes an artifact.

## Changes and preservation

Implementation changes are confined to `native_engine.py` and three publication-coordinator hooks/markers in `convert_fp16.py`. `trust.json` updates only those Python source hashes and adds the new test source. All native executable hashes, compiler identity/flags, ABI, and C sources are unchanged. There was no native rebuild or numerical optimization.

Exact pre-existing files changed: `native_engine.py`, `convert_fp16.py`, `trust.json`, `README.md`, `RESULTS.md`, `INVENTORY.md`, and `CANDIDATE_MANIFEST.json`.

New tests: `tests/test_publication_gate_v2.py` (23 methods). Existing tests are unchanged. Documentation, inventory and candidate hashes are updated. All 291 prior candidate files, including the previous manifest and evidence, are preserved byte-for-byte in `audit-history/pre-publication-gate-v2/` and verified against that archived manifest. Existing root-level `acceptance-evidence/`, `performance-summary.json`, and `integrity-results.json` describe the earlier stage; this document and the V2 evidence supersede them for current readiness.

## Regression results

The final release run passed **136 tests**, with no skips, and all **23 unchanged native fault scenarios**. It includes the exhaustive 65,536 BF16 comparisons and 585,728 single-bit corruption rejections. New regression cases cover:

1. Direct CANCEL after complete numerical verification and before publication.
2. Raw Session CANCEL, so bypassing `Native.abort()` cannot bypass the terminal state.
3. Repeated CANCEL and later successful cleanup.
4. CANCEL after validated FINISH during pre-publication fsync.
5. Replacing the first publication hook with plain successful FINISH.
6. Direct FINISH without a complete verification result.
7. Nonzero exit status and later clean STATUS, retaining failed FINISH evidence.
8. Signal-termination wait status.
9. Missing FINISH acknowledgment.
10. Incorrect FINISH worker token.
11. Incorrect FINISH PID.
12. FINISH acknowledgment without confirmed reaping.
13. FINISH acknowledgment with cleanup errors.
14. Truncated cleanup-error history.
15. Unknown wait status.
16. Separate decoding of normal, nonzero, signaled and unknown exit states.
17. Repeated successful FINISH without retransmission.
18. Cancellation during actual context-manager exit after confirmed publication.
19. Cleanup diagnostics after confirmed publication.
20. Unknown promotion outcome followed by cancellation.
21. Cancellation during interrupted CLI reporting after publication.
22. Failed pre-publication filesystem operation cannot reuse earlier authorization.
23. Session loss after FINISH denies publication while preserving confirmed reaping.

Tests inspect whether synthetic artifact directories actually exist. Workers are real supervised subprocesses. For invalid FINISH responses/nonzero-status cases, the test first lets the real supervisor complete FINISH and reap its real child, then mutates only the coordinator-visible response. This deterministically exercises the publication gate without claiming that the underlying kernel produced the injected status. Real SIGKILL cancellation and native kill/wait failures remain exercised by the other tests/scenarios.

The final 27-session validation workflow also reruns the existing bounded benchmark and standalone kernel probe; no performance changes were made. An initial expanded run passed 134 tests. A subsequent 136-test run found one incorrect test assumption: idempotent abort does not query STATUS for an already reaped child. The test now explicitly requests the later clean STATUS and verifies that it cannot repair the retained failed FINISH. The final 136-test run passes. All intermediate evidence remains available.

## Process and source integrity

Final release: **187 distinct processes with native/harness wait-confirmed reaping**, all harness sessions ending with ECHILD, and no unresolved ownership. Across initial, intermediate and final runs: **480 distinct wait-confirmed processes**, with no known PIDs remaining. The process count includes supervisors, coordinators, workers and numerical probes, not just workers. Disappearance from `/proc` is supplemental evidence; the primary evidence is native wait confirmation plus harness ECHILD.

All **206 pre-existing deployment files outside this candidate**, including the **93 accepted V3 files**, remain unchanged. The previous integration candidate, original prototype and historical evidence remain intact. All **34 authoritative model inventory/stat entries** match the historical record; the small manifest retains SHA-256 `d34c6389c0ca68df59e11f2a18f3616e7b335066fd8c493acf46044551e1daee`. No production weight payload was read or rehashed. Detailed checks are in `publication-gate-v2-integrity.json`.

## Remaining limits

The established Linux/trusted-operator assumptions remain. Native code, the compiler/loader/system libraries and Python coordinator are trusted. SIGKILL, power loss, arbitrary native corruption and unavailable scheduling/kernel service are outside unconditional guarantees. The existing V3 atomic-promotion window still reports genuinely uncertain outcomes as UNKNOWN; this correction neither changes the rename operation nor adds rollback.

The existing synthetic-size restrictions, 120-second CPU limits, 180-second session limit, startup/I/O deadlines and memory limits remain unchanged. Independent safetensors interoperability remains UNVERIFIED. No production conversion, production publication, GPU/CUDA, Docker, installation, network, AxonOS or tenant operation occurred.

Next step: independent Astra High re-audit of the gate and its evidence. Production conversion remains unauthorized.
