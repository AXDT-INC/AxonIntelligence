"""Read-only clu1 checks and exact Docker argv for the session 575 handoff."""
import argparse
import csv
import json
import os
import shlex
import socket
import subprocess
from pathlib import Path

from launch import run_args
from infer import require

IMAGE = "sha256:4d21bd834057bb3877072ae4653aa10ab5d1697c3b4fa17b41f01f791f9a8382"
GATE = "sha256:7fde6528b418d0db3c150f8c3f2c683021e92e0fc466e987b7cc9b1f174d6120"
PAIR = {"1": "GPU-85a39500-3cd6-4604-7ff4-4c07d48fb086",
        "2": "GPU-60515032-f798-5aaa-d1e4-acf58ea8bfa8"}
DOCKER = ["docker", "--host=unix:///var/run/docker.sock"]
FULL = "0,1,2,3,4,5,6,7"
HELD = "0,3,4,5,6,7"
NAME = "axonai-v100-smoke-575"
# These existing management containers can access devices. Their operator must
# serialize direct host/Docker GPU use throughout the window.
TRUSTED = {"axonos_session_launcher": "15922588b998634608480d19f6c13b97190755701e15f63e701f77533794767d",
           "buildx_buildkit_axonos-capi0": "4781f51d36b3275e8e68897c5790b67eb0476fa6a17fe286b6854dff58c97b28"}


def output(*args):
    return subprocess.check_output(args, text=True, timeout=20).strip()


def check(phase):
    require(socket.gethostname().split(".")[0] == "clu1", "Expected clu1")
    rows = list(csv.reader(output("nvidia-smi", "--query-gpu=index,uuid,memory.free",
                                  "--format=csv,noheader,nounits").splitlines(), skipinitialspace=True))
    devices = {index: uuid for index, uuid, free in rows}
    require(all(devices.get(i) == uuid for i, uuid in PAIR.items()), "Physical UUID mapping changed")
    pool = FULL if phase in ("baseline", "returned", "clear-full") else HELD
    if phase not in ("clear", "clear-full"):
        declarations = [line for line in Path("/home/cluadmin/AxonOS/.env").read_text().splitlines()
                        if line.startswith("AXGT_GPU_DEVICE_IDS=")]
        require(declarations == ["AXGT_GPU_DEVICE_IDS=" + pool], "Persistent allocation pool differs")
    require(output(*DOCKER, "exec", "axonos", "printenv", "AXGT_GPU_DEVICE_IDS") == pool,
            f"Live gate must have AXGT_GPU_DEVICE_IDS={pool}")
    require(output(*DOCKER, "inspect", "--format", "{{.Image}} {{.State.Health.Status}}", "axonos")
            == GATE + " healthy", "Gate image or readiness changed")
    require(output(*DOCKER, "image", "inspect", "--format", "{{.Id}}", IMAGE) == IMAGE,
            "Stage 1 image unavailable")
    if phase == "returned":
        return
    sql = """SELECT id,status,gpu_ids FROM axgt_sessions WHERE id=575
             OR status IN ('active','credit_grace') OR
             (status <> 'ended' AND allocation_status='allocating')"""
    result = output(*DOCKER, "exec", "axonos_postgres", "psql", "-XqAt", "-v", "ON_ERROR_STOP=1",
                    "-U", "axonos_gate", "-d", "axonos_gate", "-c", "BEGIN READ ONLY", "-c", sql,
                    "-c", "ROLLBACK")
    sessions = {sid: (status, ids.split(",")) for sid, status, ids in
                (line.split("|") for line in result.splitlines())}
    desktop = sessions.get("575") == ("active", ["1", "2"])
    if phase in ("baseline", "held"):
        require(desktop, "Session 575 must still own 1,2")
    elif phase == "ready" or not desktop:
        require(sessions.get("575", (None,))[0] == "ended", "Session 575 teardown incomplete")
    for sid, (_, ids) in sessions.items():
        if sid != "575":
            require(not PAIR.keys() & set(ids), f"Session {sid} holds a selected GPU")

    names = output(*DOCKER, "ps", "-a", "--format", "{{.Names}}").splitlines()
    require(NAME not in names, "Smoke container exists")
    require(("axgt-session-575" in names) == desktop, "Desktop teardown or ownership mismatch")
    # Inspect locally. Never print container environment or full inspect documents.
    containers = json.loads(output(*DOCKER, "inspect", *names))
    for c in containers:
        name = c["Name"][1:]
        h = c["HostConfig"]
        if name == "axonos" or TRUSTED.get(name) == c["Id"]:
            continue
        env = dict(e.split("=", 1) for e in c["Config"]["Env"] or [] if "=" in e)
        require(not h["Privileged"] and not h.get("DeviceCgroupRules") and not h.get("Devices")
                and not any(m["Source"] == "/" or m["Source"].startswith("/dev/") or m["Source"] == "/dev"
                            for m in c["Mounts"]), f"Review device access of {name}")
        requests = h.get("DeviceRequests") or []
        if name.startswith("axgt-session-"):
            sid = name.removeprefix("axgt-session-")
            require(sid in sessions and len(requests) == 1 and c["State"]["Status"] == "running",
                    f"Unmatched tenant {name}")
            require(set(requests[0].get("DeviceIDs") or []) == set(sessions[sid][1]), f"GPU mismatch {name}")
        if not requests:
            require(env.get("NVIDIA_VISIBLE_DEVICES", "") in ("", "none", "void"), f"Unscoped access {name}")
        for request in requests:
            ids = request.get("DeviceIDs") or []
            require(ids and all(d in devices or d in devices.values() for d in ids), f"Unscoped request {name}")
            if name != "axgt-session-575":
                require(not set(ids) & (set(PAIR) | set(PAIR.values())), f"GPU conflict {name}")
    if desktop and phase in ("baseline", "held"):
        return
    sample = output("nvidia-smi", "pmon", "-i", "1,2", "-c", "1")
    clients = [line.split()[:2] for line in sample.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if desktop:
        desktop_pids = set(output(*DOCKER, "top", "axgt-session-575", "-eo", "pid").split())
        require({row[0] for row in clients} == set(PAIR)
                and all(pid == "-" or pid in desktop_pids for _, pid in clients),
                "GPU client outside desktop 575 or incomplete sample")
        return
    require(sorted(clients) == [["1", "-"], ["2", "-"]], "Active GPU clients or incomplete NVML sample")
    if phase == "ready":
        require(all(int(free) >= 29830 for i, _, free in rows if i in PAIR), "Insufficient free VRAM")


def command(window):
    argv = run_args(IMAGE, list(PAIR.values()), 575)
    return DOCKER + argv[1:2] + ["--cidfile", str(window / "cid")] + argv[2:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["baseline", "held", "ready", "clear", "clear-full", "returned", "command", "execute"])
    parser.add_argument("window", type=Path, nargs="?")
    args = parser.parse_args()
    if args.phase in ("command", "execute"):
        if not (args.window and args.window.is_absolute()):
            parser.error("absolute window directory required")
        argv = command(args.window)
        if args.phase == "execute":
            check("ready")
            os.execvp(argv[0], argv)
        print(shlex.join(argv))
    else:
        check(args.phase)
        print(f"Pair 1,2: {args.phase} check passed")
