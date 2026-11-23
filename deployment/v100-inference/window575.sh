#!/usr/bin/env bash
# Operator-attended one-shot window. Production actions only in apply / restore.
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
op=${1:?expected apply, smoke or restore}
w=${2:?expected absolute private window directory}
[[ $w =~ ^/data/axonai/deployment/v100-inference/window-[A-Za-z0-9]{8}$ ]]
[[ -d $w && ! -L $w && $(stat -c '%u:%a' "$w") == "1000:700" && $(id -u) == 1000 ]]
export DOCKER_HOST=unix:///var/run/docker.sock
unset DOCKER_CONTEXT
exec 9>"$w/lock"
flock -w 150 9
envfile=/home/cluadmin/AxonOS/.env
full=0,1,2,3,4,5,6,7
held=0,3,4,5,6,7
gate=sha256:7fde6528b418d0db3c150f8c3f2c683021e92e0fc466e987b7cc9b1f174d6120
image=sha256:4d21bd834057bb3877072ae4653aa10ab5d1697c3b4fa17b41f01f791f9a8382
timer=axonai-${w##*/}-return.timer
check() { python3 -B pair575.py "$1"; }
change_pool() {
    [[ ! -L $envfile && $(stat -c '%u:%a:%h' "$envfile") == "1000:600:1" ]]
    [[ $(grep -c '^AXGT_GPU_DEVICE_IDS=' "$envfile") == 1 ]]
    grep -Fxq "AXGT_GPU_DEVICE_IDS=$1" "$envfile"
    sed -i "s/^AXGT_GPU_DEVICE_IDS=$1$/AXGT_GPU_DEVICE_IDS=$2/" "$envfile"
}
rollout() {
    local args=() names name
    [[ $(docker inspect --format '{{.Image}}' axonos) == "$gate" ]]
    [[ $(docker image inspect --format '{{.Id}}' axonos:latest) == "$gate" ]]
    names=$(docker ps --filter 'name=^/axgt-session-' --format '{{.Names}}')
    while read -r name; do
        case "$name" in
            axgt-session-555|axgt-session-571|axgt-session-572|axgt-session-574|axgt-session-575)
                args+=(--preserve-session "${name#axgt-session-}");;
            "") ;;
            *) echo "Tenant roster changed. Operator review required." >&2; return 1;;
        esac
    done <<< "$names"
    /home/cluadmin/AxonOS/scripts/deploy-production.sh --prevalidated-image "$gate" \
        --nvidia-package 580.173.02-1ubuntu1 "${args[@]}"
}
if [[ $op != restore ]]; then
    [[ ! -e $w/closed ]]
    systemctl is-active --quiet "$timer"
    deadline=$(cat "$w/deadline")
    [[ $deadline =~ ^[0-9]{10}$ ]]
    left=$((deadline - $(date +%s)))
    (( left >= 1380 && left <= 2400 ))
fi
case "$op" in
    apply)
        [[ ! -e $w/applying ]]
        check baseline
        ln -s -- "$w" active-pair-575
        touch "$w/applying"
        change_pool "$full" "$held"
        rollout
        check held
        touch "$w/applied"
        ;;
    smoke)
        [[ $(readlink active-pair-575) == "$w" && -f $w/applied && ! -e $w/attempted ]]
        touch "$w/attempted"
        # Keep the shell's flock through exit. Only this window's CID is eligible for cleanup.
        rc=0
        python3 -B pair575.py execute "$w" >"$w/smoke.log" 2>&1 || rc=$?
        echo "$rc" >"$w/exit-code"
        exit "$rc"
        ;;
    restore)
        touch "$w/closed"
        [[ ! -e $w/returned ]] || exit 0
        [[ $(readlink active-pair-575 || true) == "$w" ]] || { [[ ! -e $w/applying ]]; exit; }
        names=$(docker ps -a --format '{{.Names}}')
        if grep -Fxq axonai-v100-smoke-575 <<< "$names"; then
            cid=$(cat "$w/cid")
            [[ $cid =~ ^[0-9a-f]{64}$ ]]
            [[ $(docker inspect --format '{{.Id}} {{.Image}} {{.Name}}' "$cid") == "$cid $image /axonai-v100-smoke-575" ]]
            docker stop --time 15 "$cid"
        fi
        # A failed apply may have left the live gate on the original pool.
        live=$(docker exec axonos printenv AXGT_GPU_DEVICE_IDS)
        if [[ $live == "$held" ]]; then
            check clear
            if ! grep -Fxq "AXGT_GPU_DEVICE_IDS=$full" "$envfile"; then change_pool "$held" "$full"; fi
            rollout
        else
            check clear-full
            if ! grep -Fxq "AXGT_GPU_DEVICE_IDS=$full" "$envfile"; then change_pool "$held" "$full"; fi
        fi
        check returned
        unlink active-pair-575
        touch "$w/returned"
        ;;
    *) exit 2;;
esac
