# Publication Gate V3 — serialized cancellation and commit

This isolated synthetic candidate is ready for independent review only. Developer tests do not authorize production conversion. The accepted V3 package, earlier candidates, numerical implementation and original model are not changed.

## Defects corrected

V2 kept terminal cancellation history only in Python and separated its last authorization check from rename. A raw CANCEL after FINISH could escape the ledger. A direct CANCEL after the final check could complete before the coordinator subsequently renamed the artifact.

The native supervisor now records cancellation for the current worker token. COMMIT_BEGIN is serialized with CANCEL in its single-threaded event loop. The coordinator obtains a one-shot grant before entering the existing Python rename/fsync sequence. Cancellation received within that grant interval is explicitly pending, never completed. Python alone performs filesystem I/O and reports the observed outcome through COMMIT_RESULT.

This changes the observable cancellation contract: **request receipt is not cancellation completion**. Clients must inspect `cancellation_complete` and `commit_state`; successful transport, normal worker exit, reaping, or cleanup are insufficient. A completed cancellation after publication does not mean that the artifact was removed or never existed.

## Protocol and ordering

The established six-word little-endian control framing, private inherited descriptors, sequence checks, worker token and PID checks remain. The compatible operation extension is identified by `commit_protocol:3` in authenticated-source status replies and trust metadata; control framing ABI remains 2. A V3 coordinator rejects a supervisor without the extension. Neither the protocol nor its status JSON is a cryptographic proof against a compromised trusted supervisor.

| Operation | Payload | Supervisor action |
|---|---|---|
| CANCEL (4) | Empty | Record cancellation before cleanup/reply. Outside a commit, deny future grants for this token. During an in-progress or UNKNOWN commit, return pending/unresolved cancellation, never completion. |
| STATUS (3) | Empty | Report authoritative cancellation, FINISH and commit state, together with existing resource/exit facts. |
| COMMIT_BEGIN (7) | Empty | Grant only from idle with no cancellation, validated native FINISH, zero normal exit, confirmed reap and no cleanup errors. Record `in_progress` before sending `commit_decision:granted`. Otherwise reply denied. |
| COMMIT_RESULT (8) | Two uint32 words | Accept exactly once from in-progress. Record outcome and durability before replying. Reject invalid lengths, combinations and repeated/conflicting results. |

COMMIT_RESULT words:

- `(0,0)`: definitely not published, not durable.
- `(1,1)`: publication confirmed, durability uncertain.
- `(1,2)`: publication and durability confirmed.
- `(2,3)`: publication outcome UNKNOWN, durability unknown.

The supervisor cannot inspect the filesystem outcome. It trusts the established Python coordinator to report it. A native grant additionally depends on the Python gate having completed source/output checks and independent numerical verification. Raw COMMIT_BEGIN is not a numerical verification oracle or a sandbox against malicious coordinator code.

The linearization point for permission is the supervisor's transition to in-progress. CANCEL processed first permanently prevents a grant for that token. If COMMIT_BEGIN wins, it reserves the single publication attempt; a subsequent CANCEL cannot revoke it and is not reported completed while the outcome is unresolved. The actual filesystem publication point remains Linux `renameat2(RENAME_NOREPLACE)`. Parent-directory fsync is a separate durability point. **IPC ordering is not cross-process filesystem atomicity.**

A duplicate COMMIT_BEGIN with a fresh sequence is denied without creating a second grant or erasing the original in-progress grant. An exact sequence replay fails framing validation. Repeated CANCEL requests are idempotent requests for the same terminal cancellation; each response truthfully distinguishes pending from complete. Repeated FINISH does not retry numerical work or commit.

## State machine

```text
idle -- CANCEL --> cancellation recorded; future COMMIT_BEGIN denied
idle -- successful FINISH + COMMIT_BEGIN --> in_progress
in_progress -- CANCEL --> in_progress + cancellation pending
in_progress -- COMMIT_RESULT(0,0) --> not_published
in_progress -- COMMIT_RESULT(1,1/2) --> published (durability separate)
in_progress -- COMMIT_RESULT(2,3) --> UNKNOWN
in_progress -- coordinator EOF/failure, protocol failure, deadline --> UNKNOWN
published/not_published -- CANCEL --> complete only if resource cleanup is confirmed
UNKNOWN -- CANCEL --> unresolved; cancellation_complete remains false
```

`finish_completed`, cancellation history and commit state are separate from `reaped` and descriptor cleanup. STATUS, FINISH, cleanup and later zero-valued wait fields cannot clear cancellation. Python can fail closed on newer local failure evidence; it cannot override a native denial.

The transport runs multiple explicitly selected worker attempts for the synthetic suite. History applies to a worker token, not to all future test cases in the transport. START requires a strictly increasing token and rejects an outstanding in-progress grant. A resolved result, including an explicitly reported UNKNOWN, can retire that attempt; the old token is thereafter rejected and cannot regain eligibility. START is not a retry mechanism. Coordinator/session failure ends the transport rather than automatically starting another attempt. Historical filesystem UNKNOWN outcomes require operator investigation, never automatic conversion/publication retry.

## Crash and interruption semantics

| Boundary or failure | Observable behavior |
|---|---|
| CANCEL after FINISH, before grant | Native history denies COMMIT_BEGIN, including raw frames that bypass Python ledgers. No rename. |
| CANCEL after grant, before/during rename | Reply reports cancellation requested but not complete. The one granted attempt may proceed. |
| CANCEL after rename, before fsync/result | Still pending. Confirmed publication and uncertain durability remain separate facts. |
| CANCEL after confirmed COMMIT_RESULT | Complete if resources are clean; existing publication is preserved. No new commit. |
| Rename returns a definite error | Coordinator reports not published. Pending cancellation can complete. Staging is retained. |
| Interruption at rename return | Coordinator retains V3 UNKNOWN even if an artifact exists. UNKNOWN result never completes pending cancellation. |
| Parent fsync failure | Coordinator reports published with uncertain durability; it neither deletes nor retries the artifact. |
| Lost COMMIT_BEGIN response | No rename by coordinator. No retry. Supervisor may have granted; its outcome is UNKNOWN on session loss. |
| Lost COMMIT_RESULT response | Coordinator preserves its own observed publication/durability facts, but marks result acknowledgment/supervisor knowledge unknown. Supervisor may already have recorded the result. No retransmission. |
| Supervisor failure after grant | Coordinator may already hold a valid grant. Detection aborts remaining work; failure can race with rename. No claim that supervisor death prevents publication. No completed cancellation is fabricated. |
| Coordinator failure during grant | Supervisor reports UNKNOWN and reaps its owned children. It cannot infer filesystem publication from process exit or reaping. |
| Partial, malformed or replayed frame | No retry; terminate the control session. An active grant becomes UNKNOWN. Previously recorded terminal publication is not erased. |
| Secondary result-reporting failure | Record a diagnostic while retaining the primary exception and shared V3 publication state. |

Cancellation completion guarantees apply to trusted coordinator/supervisor code, documented protocol ordering, and a functioning kernel/scheduler. There is no durable distributed transaction log. SIGKILL, power loss, native memory corruption, malicious coordinator code or storage lying about fsync are outside unconditional guarantees. No acknowledgment can be promised after supervisor death. UNKNOWN requires independent filesystem recovery inspection under separate authorization.

## Implementation and preserved boundaries

Changed implementation: `protocol.h`, `supervisor.c`, rebuilt `supervisor`, `native_engine.py`, and two COMMIT_RESULT calls in `convert_fp16.py` (the pre-existing final gate now obtains a native grant). `trust.json` is regenerated for changed/new sources and the supervisor binary. `build_publication_v3.py` builds only that binary twice using the existing compiler hash and flags, requires identical bytes, then refreshes trust hashes. Sealed-memfd bootstrap and native acquisition/termination/reaping functions are unchanged.

The numerical C converter/verifier, Python reference conversion, native block processing, safetensors parsing, descriptor-relative model I/O, receipt isolation, staging behavior, no-overwrite rename primitive and existing durability/UNKNOWN reporting remain unchanged. No numerical or publication retries and no automatic fallback were added. The supervisor never reads/writes model files or receipts.

The unchanged synthetic-only entry restrictions remain: candidate-local fixture paths, at most 16 MiB payload plus metadata, one CPU/nice 19, 120-second CPU and 180-second session limits, existing startup/I/O deadlines and address-space limits. No production-scale authorization or performance claim follows from this correction. Independent safetensors interoperability remains UNVERIFIED; no reader was installed.

## Validation evidence

`publication-gate-v3-final/`: 156 tests, including all unchanged 136 V2 tests and 20 V3 regressions; no skips. All 23 unchanged native fault scenarios. Exhaustive 65,536 BF16 encodings and 585,728 corrupted outputs rejected. Native/harness accounting: 193 wait-confirmed processes, all sessions end with ECHILD.

`publication-gate-v3-process-release/`: 13 separately supervised crash/IPC scenarios, 39 wait-confirmed processes, no unresolved descendants. The runner independently inspects artifact existence after each harness has reaped its descendants. Cases cover supervisor/coordinator loss around grant/rename/result, discarded grant/result responses, partial frames, replayed COMMIT_RESULT frame and an invalid result acknowledgment. The historical case label `duplicate_result_response` exercises a duplicate request frame; `invalid_result_response` exercises response mutation. No test claims a mutated response was emitted by the kernel or native supervisor.

Combined final validation: 232 distinct wait-confirmed processes. The 20 unit regressions use the actual conversion and rename implementation, direct CANCEL and raw control frames before grant, after grant, inside the promotion function, after rename, after fsync and after result acknowledgment. Artifact existence and cancellation completion are checked independently. A test-only explicit result resolves the known no-rename fixture after injected result-report failure; production code never retries COMMIT_RESULT.

The process-loss runner creates an exclusive candidate-local temporary root, invokes the unchanged native subreaper, authenticates original executables through the unchanged bootstrap, and redirects only the synthetic coordinator entry point for fault injection. It targets only its own supervisor/coordinator. The outer runner cleans fixtures after reaping, including abrupt coordinator exit. This is test scaffolding, not an alternate production entry point.

Initial/intermediate evidence is retained, including an initial bootstrap rejection for group-writable new test files. Their modes were corrected to the existing trust policy; executable authentication was not weakened.

## Integrity and next step

The complete 699-file pre-change candidate is preserved in `audit-history/pre-publication-gate-v3/`, including its earlier archives and manifest. `publication-gate-v3.diff`, `publication-gate-v3-integrity.json`, the current inventory and manifest provide review evidence. All 206 protected deployment files, including 93 accepted V3 files, remain unchanged. Original model inventory/stat metadata are checked without reading weight payloads; this is not a fresh weight-content hash verification.

Next step: independent Astra High acceptance audit of V3, especially cancellation completion semantics, grant/result failure boundaries and token retirement. Production conversion remains unauthorized.
