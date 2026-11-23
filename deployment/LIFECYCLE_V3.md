# V3 CLI lifecycle correction — pending independent review

Root cause: outer CLI exceptions discarded known publication state and reset it to preflight.
The CLI now owns a state dictionary passed through conversion, including directory cleanup.
Receipt cleanup is explicit, preserving the primary failure and attaching secondary errors.
Publication true/confirmed survives later KeyboardInterrupt, SystemExit, I/O and reporting failures.
The rename syscall window remains unknown when confirmation is unavailable. No recovery path
removes, overwrites or re-promotes an artifact. Error reporting falls back to stderr without
replacing the primary state; if both streams fail, only nonzero exit can be guaranteed.
SIGKILL/power loss cannot emit Python state and still require independent artifact inspection.

Modified implementation: convert_fp16.py only. Numerical function ASTs unchanged; audit_io.py
and verify_fp16.py byte-identical to v2. Added tests/test_cli_cleanup.py. Existing boundary tests
plus new cleanup/reporting fault injection: 61 tests PASS in test-results-v3-final.json.
CLI cleanup tests use mocked publication with independently tracked observed state; existing
promotion-boundary tests use actual tiny synthetic filesystem artifacts. No production conversion.
Source manifest anchor and original stat metadata unchanged; no weight payload reads.
Earlier evidence preserved under audit-history/v2/ (prior v1 archive remains in place).
Receipt isolation and known container mount/interoperability limitations are unchanged.
