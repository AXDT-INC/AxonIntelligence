#!/bin/bash
# Newly authored execution wrapper; historical assembler remains unchanged.
set -euo pipefail
umask 077
R=/data/axonai/axonai-v1-original-assembler-replay-20261124
I="$R/inputs"
D=/data/axonai/datasets/axonai-v1
OUT="$R/execution/output"
PY=/data/axonai/.venv/bin/python
test ! -e "$OUT"
test ! -e "$R/logs/assembler.log"
test ! -e "$R/logs/exit-status.txt"
"$PY" -I -B -c 'import json,sys; from pathlib import Path; r=Path(sys.argv[1]); assert all(json.loads((r/"logs"/n).read_text())["status"]=="PASS" for n in ["preflight.json","confinement-check.json"])' "$R"
date -u +%Y-%m-%dT%H:%M:%SZ > "$R/logs/start-utc.txt"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export TMPDIR="$R/tmp"
unset PYTHONPATH PYTHONHOME
cd /data/axonai
set +e
/usr/bin/time -v -o "$R/logs/resource-usage.txt" \
  "$PY" -I -B "$R/tools/restrict_writes.py" "$R/execution" "$R/tmp" -- \
  "$PY" -B "$I/axonai_stage_final_20000_v11.py" \
  --batch001 "$D/sft/train/repaired/batch001-repaired-20261002T141416Z/batch001-5800-repaired.jsonl" \
  --pmc "$D/sft/train/pmc-batch002-7000.jsonl" \
  --production "$D/generated/20k-completion-production/run-v1/accepted-canonical.jsonl" \
  --recovered28-archive "$I/axonai-reconcile-28-20261008T155441Z.tar.gz" \
  --supersession-plan "$I/axonai-duplicate-replacement-plan-20261008T163702Z.tar.gz" \
  --previous11-archive "$I/axonai-duplicate12-generation-20261009T042944Z.tar.gz" \
  --single-archive "$I/axonai-slot06926-generation-20261009T045823Z.tar.gz" \
  --source12-archive "$I/axonai-duplicate12-source-freeze-20261009T042202Z.tar.gz" \
  --source1-archive "$I/axonai-slot06926-source-freeze-20261009T044828Z.tar.gz" \
  --output-dir "$OUT" \
  2>&1 | tee "$R/logs/assembler.log"
codes=("${PIPESTATUS[@]}")
set -e
printf 'assembler_exit=%s\ntee_exit=%s\n' "${codes[0]}" "${codes[1]}" > "$R/logs/exit-status.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$R/logs/end-utc.txt"
test "${codes[0]}" -eq 0
test "${codes[1]}" -eq 0
