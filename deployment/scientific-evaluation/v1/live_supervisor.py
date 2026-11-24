"""Trusted L40S host operator only. One dedicated container and private ledger per block."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import re
import select
import subprocess
import sys
import time

import runner as r
from live_package import verify, sha
from live_store import Ledger, canonical, private
from live_worker import MAX_LINE, failure, fields, parse, unpack
from live_backend import MODES, BASE, ADAPTER

LOAD_SECONDS = 900
ITEM_SECONDS = 180
PREFLIGHT_PEAK_BYTES = 42 * 2**30
DOCKER = ["docker", "--host", "unix:///var/run/docker.sock"]


def runtime(common):
    return r.Runtime(**{**common, "eos_ids": tuple(common["eos_ids"]),
                        "ordered_l40s_uuids": tuple(common["ordered_l40s_uuids"])})


def check_fixtures(jobs):
    r.require(len(jobs) == 2 and [unpack(j).item_id for j in jobs] == ["1"*32,"2"*32]
              and len(jobs[0]["input_ids"]) < r.CONTEXT-r.NEW_TOKENS
              and len(jobs[1]["input_ids"]) == r.CONTEXT-r.NEW_TOKENS
              and jobs[0]["options"] == jobs[1]["options"] and len(jobs[0]["options"]) == 2
              and len(jobs[0]["options"][1]) > 1, "incomplete synthetic control roster")


def validate_plan(plan):
    fields(plan, "run_id slot_ticket mode image uuids common synthetic jobs")
    mode = plan["mode"]
    r.require(mode in MODES and r.opaque(plan["run_id"]) and r.opaque(plan["slot_ticket"])
              and r.valid_digest(plan["image"]) and type(plan["synthetic"]) is bool, "invalid assignment")
    gpus = plan["uuids"]
    r.require(type(gpus) is list and len(gpus) == len(set(gpus)) == 2 and all(re.fullmatch(
        r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", gpu) for gpu in gpus), "invalid GPU reservation")
    jobs = plan["jobs"]
    r.require(type(jobs) is list and len(jobs) <= 80 and (jobs or mode == "discover"), "invalid roster")
    if mode == "discover":
        r.require(plan["synthetic"] and jobs == [] and plan["common"] is None, "invalid discovery")
    else:
        rt = runtime(plan["common"])
        rt.validate()
        r.require(rt.image_sha256 == plan["image"] and rt.ordered_l40s_uuids == tuple(gpus), "wrong runtime")
        r.require(plan["synthetic"] or mode != "disabled", "disabled control is synthetic only")
        if plan["synthetic"]:
            check_fixtures(jobs)
    ids = [unpack(j).item_id for j in jobs]
    r.require(len(ids) == len(set(ids)), "duplicate item")


def planspec(path, pin):
    private(path)
    raw = path.read_bytes()
    r.require(sha(raw) == pin, "unapproved plan bytes")
    from live_worker import pairs, reject
    spec = json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)
    validate_plan(spec)
    return spec


def mount(source, target):
    source = source.resolve(strict=True)
    r.require("," not in str(source) and "\n" not in str(source), "invalid mount path")
    return ["--mount", f"type=bind,src={source},dst={target},readonly,bind-recursive=readonly"]


def container_args(plan, root, base, adapter, name):
    # Only dedicated flat artifact directories may cross into the model namespace.
    prov = json.loads((root/"PROVENANCE.json").read_bytes())
    expected = {Path(e["path"]).name for e in prov["base_files"]
                if Path(e["path"]).parent == Path("/data/axonai/models/base/Qwen3.8-27B")}
    def flat(directory, names):
        r.require(directory and not directory.is_symlink() and {p.name for p in directory.iterdir()} == names
                  and all(p.is_file() and not p.is_symlink() for p in directory.iterdir()), "unsafe artifact mount")
    flat(base, expected)
    adapted = plan["mode"] in {"adapter", "disabled"}
    r.require(adapted == (adapter is not None), "wrong mount for role")
    mounts = mount(root, "/opt/axworker") + mount(base, BASE)
    if adapted:
        flat(adapter, {"adapter_config.json", "adapter_model.safetensors"})
        mounts += mount(adapter, ADAPTER)
    gpus = ",".join(plan["uuids"])
    return [*DOCKER, "create", "--pull=never", "-i", "--name", name, "--label", "axonai.session="+name,
            "--network=none", "--read-only", "--ipc=private", "--cap-drop=ALL",
            "--security-opt=no-new-privileges", "--user=65534:65534", "--workdir=/tmp",
            "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=512m", "--pids-limit=512",
            "--memory=128g", "--memory-swap=128g", "--shm-size=1g", "--log-driver=none",
            "--gpus", '"device='+gpus+'"', "-e", "CUDA_VISIBLE_DEVICES="+gpus,
            "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility", "-e", "PYTHONHASHSEED=42",
            "-e", "CUBLAS_WORKSPACE_CONFIG=:4096:8", "-e", "HF_HUB_OFFLINE=1", "-e", "HOME=/tmp",
            "-e", "HF_HOME=/tmp/hf", "-e", "PYTHONNOUSERSITE=1", *mounts,
            "--entrypoint=timeout", "sha256:"+plan["image"], "--signal=KILL", "4h",
            "python3", "-B", "-u", "/opt/axworker/live_worker.py"]


class WorkerFatal(Exception):
    """Worker-reported, allowlisted failure classification."""


class Channel:
    """Bounded request/response pipe. Deadline covers writes, prefill and hanging native calls."""
    def __init__(self, proc):
        self.proc = proc
        self.reader = proc.stdout.fileno()
        self.writer = proc.stdin.fileno()
        os.set_blocking(self.reader, False)
        os.set_blocking(self.writer, False)

    def exchange(self, request, seconds):
        end = time.monotonic()+seconds
        data = canonical(request)+b"\n"
        r.require(len(data) <= MAX_LINE, "request too large")
        while data:
            if time.monotonic() >= end or not select.select([], [self.writer], [], max(0, end-time.monotonic()))[1]:
                raise TimeoutError
            data = data[os.write(self.writer, data):]
        buf = bytearray()
        while b"\n" not in buf:
            if time.monotonic() >= end or not select.select([self.reader], [], [], max(0, end-time.monotonic()))[0]:
                raise TimeoutError
            chunk = os.read(self.reader, 65536)
            if not chunk:
                raise EOFError
            buf.extend(chunk)
            r.require(len(buf) <= MAX_LINE, "oversize worker reply")
        r.require(buf.count(b"\n") == 1 and buf.endswith(b"\n"), "unsolicited worker frame")
        reply = parse(buf)
        if isinstance(reply, dict) and set(reply) == {"fatal"}:
            r.require(reply["fatal"] in {"identity_or_contract","missing_dependency","host_io","model_runtime"},
                      "unknown worker failure")
            raise WorkerFatal(reply["fatal"])
        return reply


def docker(*args):
    return subprocess.check_output([*DOCKER,*args], stderr=subprocess.DEVNULL, timeout=60).decode().strip()


def cleanup(name):
    ids = docker("ps", "-aq", "--filter", "name=^/"+name+"$", "--filter", "label=axonai.session="+name)
    if ids:
        r.require(re.fullmatch("[0-9a-f]{12,64}", ids), "ambiguous container identity")
        docker("rm", "-f", ids)


def inspect_container(doc, plan, root, base, adapter):
    """Check effective engine config before the container's first instruction."""
    config, host = doc["Config"], doc["HostConfig"]
    r.require(doc["State"]["Status"] == "created" and doc["Image"] == "sha256:"+plan["image"]
              and config["User"] == "65534:65534" and not config.get("Volumes")
              and not host["Privileged"] and host["ReadonlyRootfs"] and host["NetworkMode"] == "none"
              and host["IpcMode"] == "private" and not host.get("PidMode") and not host.get("CapAdd")
              and "ALL" in host["CapDrop"] and {"no-new-privileges", "no-new-privileges:true",
                  "no-new-privileges=true"}.intersection(host["SecurityOpt"])
              and host["Memory"] == host["MemorySwap"] == 128*2**30 and host["PidsLimit"] == 512
              and host["LogConfig"]["Type"] == "none", "container isolation mismatch")
    devices = host["DeviceRequests"]
    r.require(len(devices) == 1 and devices[0]["DeviceIDs"] == plan["uuids"], "device reservation mismatch")
    expected = {"/opt/axworker":str(root.resolve()), str(BASE):str(base.resolve())}
    if adapter is not None:
        expected[str(ADAPTER)] = str(adapter.resolve())
    actual = {}
    for mount_info in doc["Mounts"]:
        if mount_info["Type"] == "tmpfs" and mount_info["Destination"] == "/tmp":
            continue
        r.require(mount_info["Type"] == "bind" and not mount_info["RW"], "unexpected writable or anonymous mount")
        actual[mount_info["Destination"]] = mount_info["Source"]
    r.require(actual == expected, "container mount source mismatch")


@contextlib.contextmanager
def container(cmd, name, plan, root, base, adapter):
    proc = None
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        inspect_container(json.loads(docker("inspect", name))[0], plan, root, base, adapter)
        proc = subprocess.Popen([*DOCKER, "start", "-ai", name], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        yield Channel(proc)
        proc.stdin.close()
        r.require(proc.wait(timeout=10) == 0, "worker shutdown failed")
    finally:
        try:
            cleanup(name)
        finally:
            if proc is not None:
                proc.kill()
                proc.wait(timeout=10)
                proc.stdin.close()
                proc.stdout.close()


def ops(plan):
    for job in plan["jobs"]:
        kinds = ["generate"]
        if job["options"]:
            kinds.append("options")
        if plan["synthetic"]:
            kinds.append("probe")
        for kind in kinds:
            yield job["item_id"]+"/"+kind, {"op": kind, "job": job}


def incident(plan, task, reason):
    rt = runtime(plan["common"])
    if task["op"] == "generate":
        row = r.run_item(plan["run_id"], plan["slot_ticket"], unpack(task["job"]), rt, rt, failure(reason))
        return {"receipt": row, "tokens_per_second": None, "decode_tokens_per_second": None}
    return {"op": task["op"], "input_sha256": r.digest(task["job"]["input_ids"]),
            "options_sha256": r.digest(task["job"]["options"]), "status": reason, "latency_s": None, "peaks": None, "value": None}


def abort(db, reason):
    db.bind("halt", reason)
    raise r.IntegrityError("block halted; review authenticated ledger")


def terminal_halt(plan, task, result):
    if task["op"] == "generate":
        row = result["receipt"]
        status, metrics = row["status"], row["measurement"]
        peaks = metrics["peak_reserved_bytes"] if metrics else []
    else:
        status = result["status"]
        peaks = result["peaks"]["reserved_bytes"] if result["peaks"] else []
    if status not in {"ok","empty_output","length_limit","context_overflow"}:
        return "technical_failure"
    if plan["synthetic"] and status in {"empty_output","context_overflow"}:
        return "incomplete_synthetic_control"
    if plan["synthetic"] and peaks and max(peaks) > PREFLIGHT_PEAK_BYTES:
        return "synthetic_capacity_limit"
    return None


def drive(plan, db, connection):
    r.require(db.get("halt") is None, "block previously halted")
    pending = []
    for name, task in ops(plan):
        claim = db.get("claim/"+name)
        done = db.get("done/"+name)
        r.require(not done or claim == r.digest(task), "unbound terminal receipt")
        if done:
            reason = terminal_halt(plan, task, done)
            if reason:
                abort(db, reason)
        elif claim is not None:
            r.require(claim == r.digest(task), "interrupted assignment changed")
            db.put("done/"+name, incident(plan, task, "worker_lost"))
            abort(db, "worker_lost")
        else:
            pending.append((name, task))
    if db.get("complete") is not None:
        r.require(not pending, "completed ledger missing result")
        return
    with connection() as chan:
        hello = {k:v for k,v in plan.items() if k != "jobs"}
        ready = chan.exchange(hello, LOAD_SECONDS)
        fields(ready, "ready fixtures")
        if plan["synthetic"]:
            check_fixtures(ready["fixtures"])
        else:
            r.require(ready["fixtures"] == [], "unexpected auxiliary items")
        identity = ready["ready"]
        r.require(identity["mode"] == plan["mode"] and identity["model"]["provenance_sha256"] == r.PROVENANCE_SHA256,
                  "unexpected model attestation")
        expected_adapter = plan["mode"] in {"adapter","disabled"}
        r.require(identity["model"]["base_revision"] == r.BASE_REVISION and
                  identity["model"]["adapter_sha256"] == ("5280cf5f30265038a6fee6aa57955b829d387b7d78fb9fe6afab86d65ac50978" if expected_adapter else None),
                  "incorrect slot model")
        runtime(identity["common"]).validate()
        if plan["synthetic"]:
            peaks = identity["load_reserved_bytes"]
            r.require(type(peaks) is list and len(peaks) == 2 and all(
                type(n) is int and 0 <= n <= PREFLIGHT_PEAK_BYTES for n in peaks), "load capacity gate failed")
            if plan["mode"] != "discover":
                r.require(max(identity["warmup"]["measurement"]["peak_reserved_bytes"]) <= PREFLIGHT_PEAK_BYTES,
                          "warmup capacity gate failed")
        r.require(identity["common"]["image_sha256"] == plan["image"] and
                  identity["common"]["ordered_l40s_uuids"] == plan["uuids"], "runtime attestation mismatch")
        if plan["mode"] != "discover":
            r.require(identity["common"] == plan["common"], "worker runtime mismatch")
            r.require(not plan["synthetic"] or plan["jobs"] == ready["fixtures"], "synthetic schedule changed")
        if db.get("ready") is None:
            db.put("ready", ready)
        else:
            old = db.get("ready")
            r.require(old["ready"]["common"] == identity["common"] and old["fixtures"] == ready["fixtures"],
                      "recovery runtime drift")
        import secrets
        db.put("boot/"+secrets.token_hex(16), {"at_ns":time.time_ns(), "identity":identity})
        for name, task in pending:
            db.put("claim/"+name, r.digest(task))
            started = time.monotonic()
            try:
                reply = chan.exchange(task, ITEM_SECONDS)
            except (EOFError, BrokenPipeError, TimeoutError) as exc:
                reason = "timeout" if isinstance(exc, TimeoutError) else "worker_lost"
                db.put("done/"+name, incident(plan, task, reason))
                abort(db, reason)
            fields(reply, "request_sha256 result")
            r.require(reply["request_sha256"] == r.digest(task), "worker replied to wrong assignment")
            res = reply["result"]
            validate_result(plan, task, res)
            events = {"done/"+name: res, "wall/"+name: time.monotonic()-started}
            reason = terminal_halt(plan, task, res)
            if reason:
                events["halt"] = reason
            db.put_many(events)
            if reason:
                raise r.IntegrityError("block halted; terminal outcome preserved")
    db.put("complete", True)


def validate_result(plan, task, result):
    import math
    job = task["job"]
    if task["op"] == "generate":
        fields(result, "receipt tokens_per_second decode_tokens_per_second")
        row = result["receipt"]
        r.require((row["run_id"], row["slot_ticket"], row["item_id"], row["input_sha256"]) ==
                  (plan["run_id"], plan["slot_ticket"], job["item_id"], r.digest(job["input_ids"])),
                  "misbound generation")
        return  # Full two-slot validation in audit_pairs after both private ledgers close.
    fields(result, "op input_sha256 options_sha256 status value latency_s peaks")
    r.require(result["op"] == task["op"] and result["input_sha256"] == r.digest(job["input_ids"])
              and result["options_sha256"] == r.digest(job["options"]), "misbound auxiliary receipt")
    if result["status"] != "ok":
        r.require(result["status"] in r.FAILURES and result["value"] is None, "invalid failure")
    elif task["op"] == "options":
        rows = result["value"]
        r.require(type(rows) is list and len(rows) == len(job["options"]), "wrong number of option scores")
        for row, ids in zip(rows, job["options"]):
            fields(row, "tokens log_probability")
            r.require(row["tokens"] == len(ids) and math.isfinite(row["log_probability"])
                      and row["log_probability"] <= 0, "invalid conditional likelihood")
    else:
        fields(result["value"], "argmax logits_sha256 input_sha256 terminal_logits_sha256 checked_context_tokens cache_reference_max_abs")
        n = len(job["input_ids"])
        r.require(result["value"]["checked_context_tokens"] == n+(r.NEW_TOKENS if n == r.CONTEXT-r.NEW_TOKENS else 1)
                  and r.valid_digest(result["value"]["terminal_logits_sha256"])
                  and r.valid_digest(result["value"]["logits_sha256"]) and
                  result["value"]["input_sha256"] == result["input_sha256"], "bad equivalence probe")


def save_plan(root, plan):
    validate_plan(plan)
    private(root, True)
    data = canonical(plan)+b"\n"
    # Publish complete bytes once. A crash before publication leaves no usable plan.
    import tempfile
    with tempfile.NamedTemporaryFile(dir=root) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        os.link(tmp.name, root/"plan.json")
    fd = os.open(root, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return sha(data)


def session(plan_pin, package_pin):
    return {"version": 1, "plan_sha256": plan_pin, "package_sha256": package_pin}


@contextlib.contextmanager
def opened(root, pin, key, package_pin, *, readonly=True):
    spec = planspec(root/"plan.json", pin)
    ledger = Ledger(root, key, session(pin, package_pin), readonly=readonly)
    try:
        yield spec, ledger
    finally:
        ledger.close()


def collect(plan, db):
    r.require(db.get("complete") is True and db.get("halt") is None, "incomplete or halted block")
    ready = db.get("ready")
    r.require(ready and ready["ready"]["common"] == plan["common"], "unbound runtime")
    outcomes = {}
    for name, task in ops(plan):
        result = db.get("done/"+name)
        r.require(result and db.get("claim/"+name) == r.digest(task), "missing authenticated attempt")
        validate_result(plan, task, result)
        outcomes[name] = result
    return outcomes


def compare(blocks, synthetic):
    modes = ["bare", "disabled", "adapter", "bare", "adapter"] if synthetic else ["bare", "adapter"]
    r.require([p["mode"] for p,_ in blocks] == modes, "incorrect comparison roles")
    first, reference = blocks[0]
    tickets = [p["slot_ticket"] for p,_ in blocks]
    r.require(len(set(tickets)) == len(tickets), "reused slot")
    primary = blocks[2 if synthetic else 1]
    items = [unpack(j) for j in first["jobs"]]
    differences = set()
    for index, (p, outcomes) in enumerate(blocks):
        r.require(p["jobs"] == first["jobs"] and p["run_id"] == first["run_id"] and p["synthetic"] is synthetic
                  and p["common"] == first["common"], "unmatched block")
        if index:
            rows = [source[i.item_id+"/generate"]["receipt"] for i in items for source in (reference, outcomes)]
            r.audit_pairs(items, rows, first["run_id"], [tickets[0], tickets[index]], runtime(first["common"]))
        if synthetic:
            baseline = primary[1] if p["mode"] == "adapter" else reference
            for item in items:
                key = item.item_id
                for kind in ("generate", "options", "probe"):
                    name = key+"/"+kind
                    v, other = outcomes[name], baseline[name]
                    if kind == "generate":
                        r.require(v["receipt"]["status"] in {"ok", "length_limit"} and
                                  v["receipt"]["output_sha256"] == other["receipt"]["output_sha256"],
                                  "synthetic generation equivalence failed")
                    else:
                        r.require(v["status"] == "ok" and v["value"] == other["value"],
                                  "synthetic numerical equivalence failed")
                if outcomes[key+"/probe"]["value"] != reference[key+"/probe"]["value"]:
                    differences.add(key)
    if synthetic:
        r.require(differences, "active adapter numerical effect unverified")


def main():
    import secrets
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    cmds = parser.add_subparsers(dest="action", required=True)
    init = cmds.add_parser("init")
    init.add_argument("root", type=Path)
    init.add_argument("image")
    init.add_argument("uuids", nargs=2)
    for action in ("run", "prepare", "audit"):
        cmd = cmds.add_parser(action)
        cmd.add_argument("--package-pin", required=True)
        cmd.add_argument("--key", required=True, type=Path)
        if action != "audit":
            cmd.add_argument("--root", required=True, type=Path)
            cmd.add_argument("--pin", required=True)
        if action == "run":
            cmd.add_argument("--base", required=True, type=Path)
            cmd.add_argument("--adapter", type=Path)
        elif action == "prepare":
            cmd.add_argument("--output", required=True, type=Path)
        else:
            cmd.add_argument("--block", action="append", nargs=2, required=True, metavar=("DIR", "PIN"))
            cmd.add_argument("--synthetic", action="store_true")
    a = parser.parse_args()
    if a.action == "init":
        print(save_plan(a.root, {"mode":"discover","image":a.image,"uuids":a.uuids,"synthetic":True,
              "common":None,"jobs":[],"run_id":secrets.token_hex(16),"slot_ticket":secrets.token_hex(16)}))
        return
    package = Path(__file__).resolve().parent
    verify(package, a.package_pin)
    if a.action == "audit":
        blocks = []
        for root, pin in a.block:
            with opened(Path(root), pin, a.key, a.package_pin) as (plan, db):
                blocks.append((plan, collect(plan, db)))
        compare(blocks, a.synthetic)
        print("Authenticated block audit passed.")
        return
    with opened(a.root, a.pin, a.key, a.package_pin, readonly=a.action != "run") as (plan, db):
        if a.action == "prepare":
            r.require(plan["mode"] == "discover" and db.get("complete") is True and db.get("halt") is None,
                      "verified discovery required")
            ready = db.get("ready")
            private(a.output, True)
            for label, mode in (("bare1","bare"),("disabled","disabled"),("active1","adapter"),
                                ("bare2","bare"),("active2","adapter")):
                root = a.output/label
                root.mkdir(mode=0o700)
                pin = save_plan(root, {**plan, "mode": mode, "common":ready["ready"]["common"],
                                     "jobs":ready["fixtures"], "slot_ticket":secrets.token_hex(16)})
                print(label, pin)
            return
        name = "axon-eval-"+plan["run_id"]+"-"+plan["slot_ticket"]
        cmd = container_args(plan, package, a.base, a.adapter, name)
        cleanup(name)  # Also reap orphaned workers when an interrupted claim prevents a new launch.
        try:
            drive(plan, db, lambda: container(cmd, name, plan, package, a.base, a.adapter))
        except Exception as exc:
            if db.get("halt") is None:
                # An integrity failure invalidates the entire freeze. Never rescore its partial rows.
                for name, task in ops(plan):
                    if db.get("claim/"+name) and not db.get("done/"+name):
                        db.put("done/"+name, incident(plan, task, "backend_error"))
                category = "load_timeout" if isinstance(exc, TimeoutError) else "integrity_or_backend_error"
                db.put("halt", exc.args[0] if isinstance(exc, WorkerFatal) else category)
            raise
    print("Private block complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit("Private block failed. Inspect the restricted host ledger.")
