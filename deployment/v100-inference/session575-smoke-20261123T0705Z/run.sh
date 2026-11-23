#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
suffix=${1:?attempt label required}
[[ $suffix =~ ^[a-z0-9-]+$ && ! -e $suffix.log ]]
pair=GPU-85a39500-3cd6-4604-7ff4-4c07d48fb086,GPU-60515032-f798-5aaa-d1e4-acf58ea8bfa8
tmp=/tmp/axonai-v100-ZPZEq5aL
p=$tmp/runtime/usr/local
scripts=$tmp/runtime/opt/inference
nvidia-smi -i "$pair" --query-gpu=timestamp,uuid,memory.used,memory.free --format=csv,noheader,nounits -lms 500 > "$suffix-vram.csv" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || :; wait "$sampler" 2>/dev/null || :' EXIT
date -u +%FT%TZ > "$suffix.start"
rc=0
docker exec --user 1000:1000 axgt-session-575 \
    /usr/bin/env -i PATH="$p/bin:/usr/bin:/bin" HOME="$tmp" TMPDIR="$tmp" LD_LIBRARY_PATH="$p/lib" \
    HF_HOME="$tmp/hf" HF_HUB_OFFLINE=1 HF_DEACTIVATE_ASYNC_LOAD=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES="$pair" NVIDIA_VISIBLE_DEVICES="$pair" \
    timeout --kill-after=10s 1190s taskset -c 0-3 nice -n 10 prlimit --as=77309411328:77309411328 -- \
    setpriv --no-new-privs "$p/bin/python3.12" "$scripts/offline.py" "$scripts/infer_session.py" \
    --model /opt/axonai-smoke-ZPZEq5aL/model --reserved-uuids "$pair" > "$suffix.log" 2>&1 || rc=$?
echo "$rc" > "$suffix.exit"
date -u +%FT%TZ > "$suffix.end"
exit "$rc"
