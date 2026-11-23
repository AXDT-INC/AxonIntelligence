# Direct V100 smoke test on clu1

## Decision

Keep desktop 575 until the running central gate and its persistent configuration both exclude physical indices 1 and 2. The claim path takes the first available indices in `AXGT_GPU_DEVICE_IDS`. Normal session release holds the allocation lock through desktop teardown. When release finishes, another claim can take either card before an independent Docker client starts. Docker UUID selection does not participate in this lock.

The verified control for this deployment is to change the gate pool from `0,1,2,3,4,5,6,7` to `0,3,4,5,6,7`, then apply it to the gate using the existing production rollout helper. Returning the pool requires a second same-image rollout. Neither rollout rebuilds an image. Gate clients can briefly disconnect at both transitions. Tenant containers, the launcher, the database and the Docker daemon stay running. New claims can use the other six GPUs, subject to the site's admission policy. The eight-GPU profile is unavailable during the hold.

The operator must approve both maintenance transitions, the session-575 release and the single inference attempt. Coordinate with any other claim authority, deployment automation and host/Docker administrators so nobody re-enables or directly uses the pair during the window. The production helper requires an external maintenance/admission window for each rollout. If the operator cannot assure that coverage or the gate reconnects are unacceptable, keep the desktop and defer. An existing independently enforceable block on **all** new claim paths through cleanup would also close the race without a rollout. No such active block was established on clu1.

## Arm and apply after approval

Run as `cluadmin` in Bash on clu1. Assign a backup operator the window path and the return deadline. The 40-minute window includes setup and the 20-minute smoke. Restoration starts immediately after the attempt or at the deadline if the terminal disconnects. Budget additional time for the return rollout and its health checks. The timer runs in the system manager as UID 1000 and only the timer registration needs operator systemd privileges.

```bash
cd /data/axonai/deployment/v100-inference
umask 077
w=$(mktemp -d "$PWD/window-XXXXXXXX")
unit=axonai-${w##*/}
date -d '+40 minutes' +%s > "$w/deadline"
sudo systemd-run --unit="$unit-return" --uid=cluadmin \
  --on-calendar="@$(cat "$w/deadline")" --timer-property=AccuracySec=1s \
  "$PWD/window575.sh" restore "$w"
systemctl list-timers --all "$unit-return.timer" --no-pager
python3 -B pair575.py command "$w"
./window575.sh apply "$w"
```

Confirm a future timer deadline and successful `held` output before ending the desktop. The apply command admits the current project desktop and validates its DB and Docker assignments, changes the one pool setting in the production checkout, and runs the reviewed rollout with the current immutable **gate** image and explicit preserved renters. It refuses unreviewed additions to the tenant roster. Keep the window path if reconnecting. An edit to the env file alone leaves the old gate active.

Save any desktop work and use AxonOS's session-specific **End session** action for **575**. Wait for teardown. If 575 expired or another owner appeared before the hold was confirmed, stop the handoff and return the pool. Avoid racing to reclaim the devices.

## Execute the single smoke after teardown

```bash
python3 -B pair575.py ready
sudo systemd-run --unit="$unit-smoke" --uid=cluadmin --wait \
  --property=RuntimeMaxSec=22min --property=TimeoutStopSec=20s \
  "$PWD/window575.sh" smoke "$w"
```

The same ready checks run again inside the launch path. They verify the excluded pool, healthy pinned gate, ended row, absence of the desktop and any other selected-card assignments, physical UUIDs, empty compute/graphics process sample, and at least 29,830 MiB free on each card. The two identified privileged management containers are trusted operator infrastructure. Operational exclusivity requires their administrators to respect the window. No CUDA probe runs on the host.

The command printed by `pair575.py command "$w"` is the exact Docker command executed after these checks. Use the systemd command above as the execution entry point so the host deadline, single-attempt check, window lock and evidence capture apply. The image entrypoint enforces 20 minutes plus a 15-second kill grace. The host service bounds the attached Docker client and survives logout. The worker receives only the two approved UUIDs, with four CPU equivalents, 72 GiB host RAM, zero additional swap, 256 PIDs, 512 MiB tmpfs, 128 MiB shared memory and a read-only model mount.

Review `$w/smoke.log` and `$w/exit-code`. A killed client may leave only the partial log and the systemd journal. The pinned image loads the model, checks finite prefill and cached greedy decode, then runs seven prompts. Preserve errors, peak VRAM, per-case timings and token output on failure. The window admits one attempt.

## Return the pair

Run this on success, OOM, timeout, failed preflight, or abandonment. A failed smoke service can return a nonzero exit code. Execute cleanup independently of that code.

```bash
./window575.sh restore "$w"
python3 -B pair575.py returned
test -f "$w/returned"
```

The same restore action is prearmed on the system timer. It serializes against the launch, closes the window to subsequent starts, and if necessary stops only a container matching the recorded CID, name and pinned smoke image. It checks teardown and GPU clients before re-admission. If the desktop was never released, it can also recognize the original desktop as sole scheduler owner of the pair. It restores the original pool with the existing gate helper, verifies its live image, health and persistent setting, and leaves the run evidence. A completed return makes the timer a no-op.

If recovery fails, keep the window under operator control. Inspect `journalctl -u "$unit-return.service" -u "$unit-smoke.service"`, both pool settings, Docker and NVML. Resolve the recorded discrepancy and complete the verified return. A new tenant roster needs operator review for the existing deployment helper. A timer cannot guarantee a successful production rollout during a host or Docker failure. After a host reboot, recover the private window directory and run the same return action before normal admissions or any new direct GPU launch. The on-disk exclusion survives reboot, while transient systemd units can be lost. Hand back the cards only after `returned` passes. No timer blindly admits an occupied GPU.

## Preparation evidence

On 2026-11-23 at 06:33 UTC, session 575 was active on 1 and 2. NVML reported 32,378 and 32,494 MiB free respectively. Xorg and xfwm4 were using GPU 1. Session 574 held 0, session 555 held 4, session 571 held 5 and 6, and session 572 held 7. The gate pool still contained all eight devices. The local gate source matched the live scheduler SHA-256. The deployment helper's read-only preflight passed with all five named runtimes preserved. The local inference image ID and its 20-minute entrypoint were verified against the Stage 1 record. These observations expire as allocations change.
