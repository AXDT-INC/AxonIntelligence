"""Launch only against an operator-approved project reservation. Dry run prints argv."""
import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import time

from infer import require, uuids

MODEL = "/data/axonai/models/merged/AxonAI-v1-27B-FP16"


def run_args(image, reserved, session):
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", image) and session > 0, "Expected immutable local image ID and project session ID")
    selected = ",".join(reserved)
    return ["docker","run","--rm","--pull=never","--name",f"axonai-v100-smoke-{session}",
            "--runtime=nvidia","--gpus",f'"device={selected}"',"--env",f"NVIDIA_VISIBLE_DEVICES={selected}",
            "--env",f"CUDA_VISIBLE_DEVICES={selected}","--env","NVIDIA_DRIVER_CAPABILITIES=compute,utility",
            "--network=none","--ipc=private","--cgroupns=private","--user=1000:1000","--cap-drop=ALL",
            "--security-opt=no-new-privileges=true","--read-only","--cpus=4","--cpu-shares=128",
            "--memory=72g","--memory-swap=72g","--pids-limit=256","--shm-size=128m",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=512m,mode=1777",
            "--mount",f"type=bind,src={MODEL},dst=/model,readonly,bind-recursive=disabled",
            image,"--reserved-uuids",selected]


def check_reservation(session, reserved):
    # DB clock enforces a margin beyond the container's own 20-minute timeout.
    # Session ID is a parsed positive integer. The caller must own this dedicated allocation.
    query = f"""SELECT gpu_ids FROM axgt_sessions WHERE id={session} AND status='active'
                AND allocation_status='allocated' AND expires_at > extract(epoch from now())+1320
                AND (hard_expires_at IS NULL OR hard_expires_at <= 0 OR hard_expires_at > extract(epoch from now())+1320)
                AND NOT EXISTS (SELECT 1 FROM axgt_sessions other WHERE other.id <> {session}
                    AND other.status IN ('active','credit_grace')
                    AND string_to_array(other.gpu_ids, ',') && ARRAY['0','1'])"""
    text = subprocess.check_output(["docker","exec","axonos_postgres","psql","-X","-qAt","-v","ON_ERROR_STOP=1",
        "-U","axonos_gate","-d","axonos_gate","-c","BEGIN READ ONLY","-c",query,"-c","ROLLBACK"], text=True)
    ids = text.strip().split(",")
    require(ids == ["0", "1"], "Expected an exclusive, active project reservation on indices 0 and 1 with sufficient lease time")
    inventory = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],text=True)
    by_id = {row[0]:row[1] for row in csv.reader(inventory.splitlines(),skipinitialspace=True)}
    require([by_id[i] for i in ids] == reserved, "Approved UUIDs differ from live scheduler assignment")
    holder = json.loads(subprocess.check_output(["docker","inspect",f"axgt-session-{session}"]))[0]
    requests = holder["HostConfig"]["DeviceRequests"] or []
    require(holder["State"]["Running"] and len(requests)==1 and set(requests[0]["DeviceIDs"]) == set(ids),
            "Live project container assignment differs from scheduler record")
    busy = subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid","--format=csv,noheader"],text=True)
    require(not set(busy.splitlines()).intersection(reserved), "Selected GPU already running compute")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image",required=True)
    p.add_argument("--session",required=True,type=int)
    p.add_argument("--uuids",required=True,type=uuids)
    p.add_argument("--print-command",action="store_true")
    a = p.parse_args()
    command = run_args(a.image,a.uuids,a.session)
    if a.print_command:
        print(shlex.join(command))
    else:
        check_reservation(a.session,a.uuids)
        print(f"Starting approved project session {a.session} at {time.time():.0f}",flush=True)
        os.execvp("docker",command)
