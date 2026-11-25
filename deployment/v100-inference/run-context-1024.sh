#!/usr/bin/env bash
# Run in a prepared, project-owned AxonOS Medium session from its Docker host.
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
if [[ $# -lt 2 || $# -gt 3 || ! $1 =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: bash run-context-1024.sh SESSION_ID GPU_UUID_1,GPU_UUID_2 [1024|2048|4096|sustained]" >&2
    exit 2
fi
case ${3:-1024} in
    1024|2048|4096|sustained) profile=${3:-1024} ;;
    *) echo "Unknown session test: $3" >&2; exit 2 ;;
esac
session=axgt-session-$1
pair=$2
tmp=${AXGT_RUNTIME_STAGE:?expected session-local runtime stage}
model=${AXGT_MODEL_PATH:-/home/aXonian/axonai-v100-fp16/model}
p=$tmp/runtime/usr/local
history_dir=session575-smoke-20261123T0705Z
work=$(mktemp -d "${TMPDIR:-/tmp}/axonai-$profile.XXXXXXXX")
echo "Run log and GPU samples: $work"
baseline=67cb9147e00c9fd4dce92d185c1b51405695d13a:deployment/v100-inference
git show "$baseline/infer.py" > "$work/infer_session.py"
git show "$baseline/placement.py" > "$work/placement.py"
cp score_cases.py seven-prospective.json "$history_dir/offline.py" "$work/"
patch --batch --forward --fuzz=0 "$work/infer_session.py" < "$history_dir/pread.patch"
echo "70ebf4f5f2011573e0805280bedd9e10bc65bb003af1a39745b03b0c882f19d9  $work/infer_session.py" | sha256sum -c -
patch --batch --forward --fuzz=0 -d "$work" -p1 < context-1024.patch
if [[ $profile != 1024 ]]; then
    patch --batch --forward --fuzz=0 -d "$work" -p1 < context-4096.patch
    if [[ $profile != 4096 ]]; then
        sed -i 's/^INPUT = 4096$/INPUT = 2048/' "$work/placement.py"
        sed -i 's/valve_log_4096/valve_log_2048/g; s/Expected exactly 4096/Expected exactly 2048/g' "$work/infer_session.py"
    fi
    if [[ $profile == sustained ]]; then
        patch --batch --forward --fuzz=0 -d "$work" -p1 < sustained-requests.patch
    fi
fi
cd -- "$work"
suffix=$profile
docker exec --user 1000:1000 "$session" test -x "$p/bin/python3.12"
docker exec --user 1000:1000 "$session" test -r "$model/SHA256_MANIFEST.json"
scripts=$(docker exec --user 1000:1000 "$session" mktemp -d /tmp/axonai-context.XXXXXXXX)
tar -cf - infer_session.py placement.py offline.py score_cases.py seven-prospective.json |
    docker exec -i --user 1000:1000 "$session" tar -xf - -C "$scripts"
nvidia-smi -i "$pair" --query-gpu=timestamp,uuid,memory.used,memory.free --format=csv,noheader,nounits -lms 500 > "$suffix-vram.csv" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || :; wait "$sampler" 2>/dev/null || :' EXIT
date -u +%FT%TZ > "$suffix.start"
rc=0
docker exec --user 1000:1000 "$session" \
    /usr/bin/env -i PATH="$p/bin:/usr/bin:/bin" HOME="$tmp" TMPDIR="$tmp" LD_LIBRARY_PATH="$p/lib" \
    HF_HOME="$tmp/hf" HF_HUB_OFFLINE=1 HF_DEACTIVATE_ASYNC_LOAD=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES="$pair" NVIDIA_VISIBLE_DEVICES="$pair" \
    timeout --kill-after=10s 1190s taskset -c 0-3 nice -n 10 prlimit --as=77309411328:77309411328 -- \
    setpriv --no-new-privs "$p/bin/python3.12" "$scripts/offline.py" "$scripts/infer_session.py" \
    --model "$model" --reserved-uuids "$pair" > "$suffix.log" 2>&1 || rc=$?
echo "$rc" > "$suffix.exit"
date -u +%FT%TZ > "$suffix.end"
exit "$rc"
