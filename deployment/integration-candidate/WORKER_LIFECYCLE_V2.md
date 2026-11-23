# Worker lifecycle correction V2 — pending independent re-audit

This revision fixes the two worker lifecycle findings from the independent integration audit. It does not authorize real conversion, change numerical processing, or remeasure performance. Historical benchmark results remain historical measurements of the preceding candidate.

## Root causes and corrections

1. Acquisition previously assigned the return value of Popen before entering the cleanup guard, and registered the engine context outside that guard. Interruption could strand a live child. The supervisor is now allocated and retained before initialization; its Popen object is allocated and owned before Popen.__init__ can create a child. Authentication descriptors, pipe ends and the executable memfd are registered before subsequent fallible work. Parent streams take ownership only after registration. SIGINT is temporarily deferred during acquisition/ownership transfer, with restoration inside the outer BaseException guard. This was tested using a real SIGINT during launch, not just exceptions from mocks.
2. abort() previously set closed=True unconditionally and cleared diagnostics on every invocation. closed is now derived from confirmed process status, closed pipes, an empty descriptor ledger and restored context. Cleanup diagnostics accumulate, and a completed abort is idempotent. UUID cleanup handles cannot accidentally identify a later worker through object-address reuse.

The native executable, worker.c, convert.c, verify.c, prototype.h, ABI, wire framing, input/output semantics, trust hashes, compiler flags and memfd seals are unchanged. Authentication uses the same descriptor-relative/no-follow checks, regular-file/single-link checks, ownership/mode constraints, size limits and hash comparisons. Its descriptor acquisition and closure were rewritten narrowly to make all authentication descriptors owned and to preserve primary exceptions when cleanup also fails. There is no pathname-only executable launch.

## State and ownership model

The strong OWNED registry retains each active or incompletely cleaned supervisor. A report includes worker_id, pid, process_state, state_history, unusable, reaped, returncode, individual pipe states, pending descriptors, context_pending, cleanup_complete and cumulative cleanup_errors.

- not_launched: no child is known to have been created.
- running_supervised: a live child is owned.
- unusable_failed: recorded failure transition; no further numerical operations are allowed.
- termination_requested: a termination operation is being attempted.
- termination_attempted_unconfirmed: termination or waiting has not established reaping; process status may be unknown.
- exited_reaped: wait/poll has confirmed child exit and reaping.
- cleanup_incomplete: failure occurred and at least one process/resource/context obligation remains. An exited child can still have incomplete pipe or descriptor cleanup.

A kill request is never evidence of reaping. Failed kill/wait/close operations retain operation, exception type and message. A second abort preserves earlier diagnostics. retry_cleanup(worker_id) performs only bounded resource cleanup; it never retries numerical requests or publication. A stale completed handle is rejected. Context restoration is attempted at most twice in its owning context; unresolved restoration remains explicitly reported and owned. Resource cleanup cannot repair an arbitrary failure of the context machinery in another execution context.

Pipe creation is explicit and descriptor-owned before process launch. Authentication descriptors and memfd identity are tracked to avoid blindly closing a reused descriptor number. Closed descriptors are recognized even when a close operation completed before an injected interruption. Unresolved descriptor identity is retained rather than treated as successful closure.

## Publication and error reporting

The accepted filesystem and publication operations remain Python operations. audit_io.py is byte-identical to V3. Receipt policy, source reads, staging, write loops, renameat2(RENAME_NOREPLACE), and fsync sequencing are unchanged. The existing pre-publication worker completion requirement remains in force. A broken worker cannot finish successfully even if a later cleanup retry reaps it.

Minimal reporting-only changes to candidate convert_fp16.py and verify_fp16.py attach worker_lifecycle to failure reports. They do not change promoted, durability, phase or the primary conversion/verification exception. The context manager preserves the primary exception and adds secondary context-cleanup failures. An interruption after successful publication retains confirmed promotion/durability; an interrupted promotion boundary remains unknown. No error path deletes, overwrites or re-promotes artifacts.

## Validation

Final complete suite: **133 PASS, zero failures/errors/skips**. This is the prior 112 tests plus 21 targeted lifecycle tests. The suite retains exhaustive 65,536-encoding C/V3 comparison, IPC equivalence, signed-zero/subnormal tests, independent verification, receipt isolation, collisions, ENOSPC, promotion interruptions, fsync and post-promotion failures.

New cases cover interruptions before/after actual Popen, a real SIGINT during launch, context registration before/after set, context exit, authentication read/close failures, memfd cleanup, partial pipe creation, parent-stream wrapping, kill/wait/pipe-close faults, multiple simultaneous cleanup faults, repeated abort, stale handles, unknown process status, primary-error/reporting-error preservation and cleanup after confirmed publication.

The suite runner tracked **93 real child processes**, including workers and synthetic compiler/test subprocesses. Every tracked process was confirmed reaped. No unexpected live children or unresolved ownership entries remained after the suite. New fault tests independently inspect Popen status and PID disappearance and provide unconditional harness cleanup after restoring injected operations. No assertion relies exclusively on a fake process object.

See test-results-worker-lifecycle-v2.json, lifecycle-v2-acceptance-tests.txt, worker-lifecycle-v2-integrity.json and worker-lifecycle-v2.diff. Earlier failing development runs are retained; no tests were skipped to achieve the final result.

## Exact implementation scope

Modified implementation: native_engine.py, candidate.py, convert_fp16.py, verify_fp16.py.
New implementation helper: lifecycle_support.py.
New regression source: tests/test_worker_lifecycle_v2.py.
New full-suite runner: run_tests_lifecycle_v2.py.
Documentation/inventory: README.md, this document, current CANDIDATE_MANIFEST.json, integrity/diff/test receipts and development logs.
The complete preceding 39-file candidate is preserved under history/pre-worker-lifecycle-v2/, including its original manifest and audit evidence. Original top-level historical test/benchmark reports are retained.

## Limits and review gate

Linux/CPython and a single coordinator execution context remain assumptions. SIGINT deferral is narrowly used for ownership transfer and cleanup, not a general signal-isolation mechanism. There is no guarantee against SIGKILL, power loss, fatal process termination, interpreter failure or arbitrary corruption of Python/OS internals. Bounded waits may leave cleanup explicitly incomplete if the OS cannot kill or reap a process. Such a state requires operator attention; it is not success and never authorizes promotion. In-process registry ownership is not a durable recovery service surviving coordinator death.

No production payloads were read. Synthetic path/size and worker resource limits remain unchanged. No GPU, Docker, installation, network, AxonOS or scheduler operations occurred. No performance optimization was made or benchmark extrapolation updated. Independent safetensors/runtime interoperability remains outside this correction.

Ready for independent Astra High worker-lifecycle re-audit. Stop before any actual conversion or replacement of V3.
