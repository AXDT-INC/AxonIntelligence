# Atomic native publication boundary — independent review required

A Python eligibility check followed by a ctypes call was insufficient: trace/signal-driven lifecycle code could invalidate eligibility before the actual syscall. The candidate now uses an opaque native record and one pthread mutex per publication attempt. `publication_gate.c` owns final authorization, invalidation and `renameat2(RENAME_NOREPLACE)`; it is loaded into the Python coordinator, not the supervisor. Thus only the coordinator still performs model filesystem operations.

## Native ownership and ordering

Each record has immutable session identity (coordinator PID), worker generation, eligible/invalidated flags, syscall-attempted flag, confirmed-publication flag and saved errno. Duplicate session/generation creation is rejected; invalidation never resets a record, and no record is recycled. Allocation is bounded to 4096 records per loaded image, retained until process exit. This intentionally avoids destructor/reuse races during asynchronous cleanup. Wrong identity returns ESTALE without modifying another record.

After the existing supervisor grants COMMIT_BEGIN, Python arms its native record once. Arming fails for an invalidated, previously armed or attempted record. Premature completion and lifecycle mismatch invalidate the active worker's own record before raising; the supplied mismatched object never selects the invalidation target. Terminal settling also invalidates further promotion without clearing confirmed publication.

Both `gate_invalidate` and `publish_once` acquire the same record mutex. The final eligibility decision and the rename syscall execute under that lock, with no return to Python or Python callback between them. `publish_once` consumes eligibility before the syscall, records success or errno before unlocking, and never retries. Invalidation ordered first prevents syscall entry. Promotion ordered first may publish; subsequent invalidation preserves the confirmed result and prevents any additional attempt.

The image is SHA-256 checked against trust metadata, copied into a memfd, sealed (WRITE/GROW/SHRINK/SEAL verified), and loaded through that descriptor. Production coordinator calls use PyDLL, retaining CPython's GIL. Python signal handlers and tracing callbacks therefore execute outside the native critical section. A hook immediately before native entry can invalidate; the native decision observes it. A handler delivered after return cannot retroactively undo the syscall. Mutex serialization also applies to the test-only CDLL concurrent callers that release the GIL. No native signal handler calls these APIs; no native function calls Python or waits on a Python callback.

This is local serialization, not cross-process filesystem atomicity. Supervisor COMMIT_BEGIN/CANCEL ordering is unchanged. CANCEL after a supervisor grant remains pending until the commit result resolves; it is not itself a revocation of that granted interval. A rejected completion separately invalidates local promotion. Native success is not a durability receipt: parent-directory fsync remains in the established Python sequence.

## Result handling and limitations

Native status preserves syscall-attempted, confirmed and errno facts. If Python is interrupted after native success but before recording the return, the existing conservative publication state can remain UNKNOWN; the separate native status still records confirmation. SIGKILL/power loss can lose all in-process state. There is no new durable transaction log and no automatic rename, conversion or result retry.

The existing Python target binding and descriptor-relative directory access remain. The C API is not a security sandbox against malicious trusted Python passing arbitrary pointers, paths or calling internal arming functions. Raw forged coordinator COMMIT_RESULT reports remain outside the accepted trusted-operator model. Compiler, libc, pthreads, loader, CPython and kernel are trusted. This change adds trusted native code inside the coordinator; memory corruption there could affect coordinator integrity. The build uses existing GCC, warnings-as-errors, immediate binding and RELRO; identical repeated builds are checked, not claimed reproducible across arbitrary toolchains.

No native supervisor acquisition, numerical routines, worker binaries or accepted V3 files are changed. Existing model I/O, receipt isolation, staging, no-overwrite behavior and durability policy remain. Synthetic size, CPU, memory and session limits remain; production-scale changes require separate authorization. Safetensors interoperability remains UNVERIFIED.

## Evidence and scope

The workspace already contained an unpublished PyDLL permission-word draft and tests when this task resumed. Its actual 3490 files, including existing reports and stale manifest, were preserved in `audit-history/pre-atomic-native/`; SNAPSHOT.json records actual source hashes. Nothing is represented as having matched that stale manifest. The earlier manifest-backed accepted candidates remain preserved inside that history.

`build_atomic_publication.py` builds only publication_gate.so twice with the recorded installed compiler and exact flags, compares bytes and refreshes source trust hashes. `atomic-native-release/` contains the final complete test and native process-accounting evidence. The prior draft's trace tests reproduce mismatched completion, post-Python-entry rejection and immediate-pre-native-entry rejection; each checks artifact absence. Post-rename/durability tests check actual artifact presence and preserved publication.

Six new tests cover deterministic invalidation-first and promotion-first orderings (Events and Barriers, no sleeps), concurrent contenders (either serialized outcome accepted, matched against disk), duplicate generations/wrong identities, interruption after native success, and native syscall failure with no retry. Concurrent tests repeat eight times; native status is compared with actual artifact existence. Existing unit tests, 23 native fault scenarios, 13 crash/IPC scenarios, all 65536 encodings and 585728 corruption rejections are rerun. Destructive raw-forgery regression remains separately supervised, never disabled.

See atomic-native.diff, atomic-native-integrity.json, INVENTORY.md and CANDIDATE_MANIFEST.json. This is developer validation only. Next step is independent acceptance audit; production conversion remains unauthorized.
