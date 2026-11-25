#!/usr/bin/env bash
set -euo pipefail

tool=$(cd "$(dirname "$0")" && pwd)
release=$(dirname "$tool")
repo=$(cd "$release/../.." && pwd)
model="$repo/models/merged/AxonAI-v1-27B"
cache=${AXONAI_RELEASE_CACHE:-"$release/.cache"}
stage=${AXONAI_RELEASE_STAGE:-"$release/outputs/bf16-final"}
image=${AXONAI_RELEASE_IMAGE:-axonai-hub-cpu:rc1}
args=(--rm --pull=never --runtime=runc --user=1000:1000 --read-only --cap-drop=ALL
      --security-opt=no-new-privileges --cpus=1
      --tmpfs=/tmp:rw,nosuid,nodev,noexec,size=96m -v "$tool:/tool:ro")
action=${1:?expected fetch, reconcile, prepare, verify or upload}
shift
if [[ $action == upload ]]; then
    args+=(--memory=4g --pids-limit=128 -e RAYON_NUM_THREADS=2 -e TOKIO_WORKER_THREADS=2)
else
    args+=(--memory=768m --pids-limit=64)
fi
case "$action" in
  fetch|reconcile)
    mkdir -p "$cache/pmc" "$cache/crossref" "$release/attribution"
    args+=(-v "$model/SOURCE_ATTRIBUTIONS.inherited-draft.json:/input/register.json:ro")
    if [[ $action == fetch ]]; then
      exec docker run "${args[@]}" --network=bridge -v "$cache/pmc:/cache/pmc:rw" "$image" python /tool/fetch.py
    fi
    net=none
    [[ ${1:-} != --publisher ]] || net=bridge
    exec docker run "${args[@]}" --network="$net" -v "$cache/pmc:/cache/pmc:ro" \
      -v "$cache/crossref:/cache/crossref:rw" -v "$release/attribution:/attribution:rw" \
      -v "$repo/engineering-maintenance/frozen-v1-training-extraction-v1-20281029/EXTRACTION_RECEIPT.json:/input/extraction.json:ro" \
      "$image" python /tool/reconcile.py "$@"
    ;;
  prepare|verify)
    mkdir -p "$stage"
    mode=ro
    [[ $action != prepare ]] || mode=rw
    exec docker run "${args[@]}" --network=none -v "$model:$model:ro" -v "$stage:/stage:$mode" \
      -v "$release/attribution:/attribution:ro" -v "$repo/LICENSE:/input/SOURCE_LICENSE.txt:ro" "$image" python /tool/package.py "$action" "$model"
    ;;
  upload)
    # The reusable uploader accepts one reviewed candidate and one model mount, of any precision.
    candidate=${1:?supply candidate folder}
    source_model=${2:?supply absolute source model path}
    repo_id=${3:?supply existing approved model repository}
    manifest_sha=${4:?supply approved manifest SHA-256}
    token=${HF_RELEASE_TOKEN_FILE:?supply a separate read-only token file}
    [[ $repo_id =~ ^[A-Za-z0-9_-]+/[A-Za-z0-9][A-Za-z0-9._-]*$ && $manifest_sha =~ ^[a-f0-9]{64}$ ]]
    [[ $token == /* && $candidate == /* && $source_model == /* ]]
    [[ -f $token && -d $candidate && -d $source_model ]]
    upload_cache="$cache/hub/$repo_id/$manifest_sha"
    mkdir -p "$upload_cache"/{state,hf}
    exec docker run "${args[@]}" --network=bridge -v "$source_model:$source_model:ro" \
      -v "$candidate:/candidate:ro" -v "$upload_cache/state:/candidate/.cache:rw" \
      -v "$upload_cache/hf:/cache/hf:rw" -v "$token:/run/secrets/hf_token:ro" \
      "$image" python /tool/upload.py "$repo_id" "$manifest_sha" "$source_model"
    ;;
  *) echo "Unknown action" >&2; exit 2 ;;
esac
