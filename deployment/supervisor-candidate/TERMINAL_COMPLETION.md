# Coordinator terminal-completion correction

Developer validation only; independent acceptance is required. No production conversion is authorized.

The normal Python completion API now requires the exact bound lifecycle object and a terminal result latched by the coordinator after its publication operation returns or unwinds. A premature call raises, sends no COMMIT_RESULT, and permanently blocks entry into the pending promotion operation. The native commit interval therefore remains in progress; CANCEL cannot complete merely because the rejected helper was called.

The coordinator records `commit_not_requested`, `promotion_pending` (after the native grant), `rename_attempted`, `promotion_confirmed`, `durability_confirmed` or `durability_failed_or_unknown`, `not_published` or `promotion_outcome_unknown`, and `commit_result_finalized`. The existing `commit_granted` fact remains separately recorded. Terminal outcomes are immutable tuples: not published `(0,0)`, published/uncertain `(1,1)`, published/durable `(1,2)`, or unknown `(2,3)`.

`perform_promotion` consumes the pending coordinator phase once before calling the unchanged rename primitive. It refuses a blocked, terminal or previously attempted operation. The completion helper cannot finalize while that operation or its durability work is pending. The success path latches a terminal outcome after fsync; the failure path latches only after stack unwinding has left the promotion/durability sequence. Neither path returns to promotion. Identical confirmed completion is idempotent; a different lifecycle or changed outcome fails. A lost result is not retried. Existing publication and UNKNOWN facts remain authoritative.

The new Python transition helpers are part of the trusted coordinator implementation, not a sandbox against arbitrary Python code mutating its internal state or deliberately invoking internal settling transitions out of order. A premature completion before promotion entry also blocks entry when its exception is swallowed. If execution has already entered the rename operation, rejecting a premature completion cannot undo a syscall already performed; it still cannot release the native interval. Outcome and durability uncertainty must be preserved.

No native supervisor changes were necessary. Its cancellation/commit ordering, numerical code, acquisition, reaping and executable authentication mechanisms are unchanged. `convert_fp16.promote`, source I/O helpers and the accepted baseline remain unchanged. Source hashes in `trust.json` change for the two Python implementation files and the new regression test; no executable hash changes.

## Tests and evidence

`terminal-completion-final/` contains the final unit and native scenario evidence. All 156 prior unit tests remain unchanged. Six new tests exercise:

- Premature completion followed by raw CANCEL: no completed cancellation before terminal unwind; final artifact absent.
- Swallowed premature-completion exception followed by attempted rename: promotion guard rejects; artifact absent.
- Identical repeated completion: no additional frame; finalized promotion cannot be entered again.
- Conflicting terminal outcome: rejected without changing the actual published artifact.
- Different lifecycle object: rejected.
- Deliberately forged raw COMMIT_RESULT: remains outside the trusted coordinator model and can still cause premature native nonpublication evidence; actual artifact existence is checked, and subsequent normal result delivery fails rather than claiming success.

The raw-forgery case intentionally terminates its transport when the later genuine result conflicts. `run_terminal_tests.py` executes that unchanged test in a fresh native-supervisor session and the other 161 tests in another. It asserts that executed IDs equal the full discovery set, with no skips or duplicates. No test was renamed or disabled. The initial shared-session test run is retained; it showed that the intentional channel termination disrupted three later tests. Automatic approval review rejected a proposed test-method rename; the final solution keeps every test and isolates the destructive session instead.

The existing 23 native fault scenarios, 13 crash/IPC scenarios, exhaustive 65,536 BF16 comparisons and 585,728 corruption rejections are rerun. Native wait events and harness ECHILD establish reaping, not just disappearance from process listings. Test fixtures are candidate-local and cleaned after execution.

## Trust boundary and limitations

Raw, well-formed forged COMMIT_RESULT frames from the designated coordinator channel are not independently verified against the filesystem by the supervisor. The raw-forgery test deliberately demonstrates this limitation; it is not represented as a security success. Protecting against a malicious coordinator requires a separately reviewed ownership/capability or filesystem-witness architecture, outside this change. The ordinary coordinator never treats a conflicting response as a new grant or retries promotion.

SIGKILL/power loss, trusted loader/system libraries, synthetic CPU/session/size limits and unverified independent safetensors interoperability remain as documented in PUBLICATION_GATE_V3.md. No unconditional protection against arbitrary trusted-code corruption is claimed.

The pre-change 1,594-file candidate is archived in `audit-history/pre-terminal-completion/`. See `terminal-completion.diff`, `terminal-completion-integrity.json`, INVENTORY.md and CANDIDATE_MANIFEST.json for review.
