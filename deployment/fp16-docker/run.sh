#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
source_model=/data/axonai/models/merged/AxonAI-v1-27B
destination=/data/axonai/models/merged/AxonAI-v1-27B-FP16
image=$(cat image.id)
[[ "$image" =~ ^sha256:[0-9a-f]{64}$ ]]
[[ -d "$destination" && ! -L "$destination" && -d "$source_model" && ! -L "$source_model" ]]
case "${1:-fresh}" in
  fresh) first=$(find "$destination" -mindepth 1 -maxdepth 1 -print -quit)
         [[ -z "$first" ]] || { echo 'Fresh run requires an empty destination' >&2; exit 1; } ;;
  resume) [[ -d "$destination/.conversion" ]] || { echo 'Missing interrupted conversion' >&2; exit 1; } ;;
  *) echo 'Use fresh or resume' >&2; exit 2 ;;
esac
docker run --rm --pull=never --name axonai-v1-fp16-convert \
  --runtime=runc --platform=linux/amd64 --network=none --ipc=private --cgroupns=private \
  --user=1000:1000 --cap-drop=ALL --security-opt=no-new-privileges=true --read-only \
  --cpus=2 --cpu-shares=128 --memory=16g --memory-swap=16g --pids-limit=64 \
  --shm-size=32m --tmpfs=/tmp:rw,noexec,nosuid,nodev,size=128m,mode=1777 \
  --env NVIDIA_VISIBLE_DEVICES=void --env CUDA_VISIBLE_DEVICES= \
  --mount "type=bind,src=$source_model,dst=/source,readonly,bind-recursive=disabled" \
  --mount "type=bind,src=$destination,dst=/output,bind-recursive=disabled" \
  "$image"
