#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
image=axonai-v1-fp16-cpu:torch2.13.0-r2
existing=$(docker image ls --quiet --filter "reference=$image")
[[ -z "$existing" && ! -e image.id ]] || { echo 'Dedicated image tag or image.id already exists' >&2; exit 1; }
docker buildx build --builder default --platform linux/amd64 --load \
  --iidfile image.id --tag "$image" .
