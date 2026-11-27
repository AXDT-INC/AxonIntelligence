# Private GPQA Diamond extension

`gpqa.py` adds GPQA to an existing L40S FP16 comparison without changing its frozen
worker code, roster, ledger, checkpoints or completed outcomes. It lives outside
the worker directory because the running controller hashes that directory before
each block. Do not modify frozen worker Python or Compose files during collection.

The 27 November 2026 extension is under the private, Git-ignored directory
`receipts/l40s-benchmark-20261127/gpqa-extension/`. Access was authenticated against
the pinned `Idavidrein/gpqa` revision
`83022cefff930aea54f654c0b282e74b9eeda5c6`. Source contents, questions, choices,
private option permutations, answer keys, raw responses and attempt records must
remain there; do not include them in repository files, public logs or reports.
The directory is mode 0700 and its data files are mode 0600.

Preparation freezes 198 unique record IDs and sorts the four choices by SHA-256
of UTF-8 `42/gpqa-<Record ID>/<original choice index>` (original index zero is the
correct choice). Both arms receive identical token IDs. One question has 2,803
input tokens and exceeds the unchanged 2,048-token limit, leaving 197 eligible
questions and 394 scheduled responses. Outputs retain the 256-token limit.
JSONL escapes Unicode separators without changing the decoded prompt contents.
The two blocks contain 99 planned questions each and run Qwen/AxonAI then
AxonAI/Qwen. Existing frozen SFT and staging corpora were screened before dispatch;
incomplete exposure and family coverage still preclude a clean holdout claim.

The durable queue waits for the original `completion.json`, then obtains the same
two GPU locks and checks qualification and code hashes before loading a model.
It uses the original claim-before-dispatch ledger and never reruns claimed slots.
It verifies hashes of previously completed parent rows without writing to that
ledger. The worker receives prompts only; source files and gold keys stay on the
host outside the container. No dataset downloads occur inside inference workers.

Inspect `queue-status.json` and `execution.json` for queue state and process ID.
Only after confirming no existing queue process is alive, a stopped extension can
be resumed with:

```bash
.venv/bin/python -Bu deployment/scientific-evaluation/public-extensions/gpqa.py \
  queue receipts/l40s-benchmark-20261127
```

Keep the process supervised or detached for long execution. Do not repeat
`prepare` against an already frozen extension. If the original phase stops before
completion, this queue continues waiting; resolve the original phase using its
existing ledger before resuming collection. Avoid editing frozen controller code.

After collection, the extension writes its own `results.json` and `completion.json`,
then parent-level `combined-results.json` and `combined-completion.json`. Parent
results remain unchanged. GPQA is descriptive; PubMedQA remains the primary
contrast. Aggregate reports include eligibility and missingness rather than
treating the excluded question as an executed example. Total planned coverage is
6,200 questions, 6,199 eligible and 12,398 response slots.

Synthetic tests cover deterministic choices, Unicode JSONL round trips, detection
of changes to completed parent rows, and additive analysis without overwriting
the original result:

```bash
.venv/bin/python -B -m unittest discover \
  -s deployment/scientific-evaluation/public-extensions -p 'test_*.py' -v
```

## Aggregate progress and completion estimates

Read current counts, inference and format failures, cap hits, throughput and
provisional completion estimates without scoring accuracy or emitting item IDs,
examples, keys or raw responses:

```bash
.venv/bin/python -B deployment/scientific-evaluation/public-extensions/status.py \
  receipts/l40s-benchmark-20261127
```

The report reads SQLite in read-only mode. It projects the remaining block order
from observed suite/arm latency, uses the qualification stress ratio for an arm
without observations, and estimates unobserved GPQA at the full output cap with
the qualification stress latency as a floor. It allows 150 seconds per future
model load. Estimates exclude outages and scoring time and are provisional until
both arms have representative observations. Format failures and cap hits can
overlap and must not be summed as distinct failures.

The current controllers were launched with a new session, stdin connected to
`/dev/null`, and stdout/stderr connected to private files. They are reparented to
PID 1 and have no controlling terminal. This host's logind reports
`KillUserProcesses=false`; the originating SSH session is already abandoned.
Closing the IDE therefore leaves collection and automatic scoring running. This
is process detachment, not reboot recovery or automatic restart after a fatal
error. Keep the host and Docker running, and preserve the frozen code and GPU
assignment. `benchmark_run.collect` invokes scoring and writes completion after
the original collection; the GPQA queue then collects, scores and combines both
aggregate reports without requiring an IDE or another interactive command.
