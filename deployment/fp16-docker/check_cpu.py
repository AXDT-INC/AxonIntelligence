import os
import socket
import sys
from pathlib import Path


def check_devices():
    devices = Path("/dev")
    if (list(devices.glob("nvidia*")) or any((devices / n).exists() for n in ("dri", "kfd", "dxg")) or
            os.environ.get("NVIDIA_VISIBLE_DEVICES") != "void" or os.environ.get("CUDA_VISIBLE_DEVICES") != ""):
        raise RuntimeError("GPU isolation check failed before importing torch")


def check_packages():
    import torch
    import numpy
    import safetensors
    if (torch.version.cuda is not None or torch.version.hip is not None or
            torch.__version__ != "2.13.0+cpu" or numpy.__version__ != "2.2.6" or safetensors.__version__ != "0.8.0"):
        raise RuntimeError("Unexpected CPU build or package version")


def check_runtime():
    if (os.getuid(), os.getgid()) != (1000, 1000):
        raise RuntimeError("Expected UID/GID 1000:1000")
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    if int(status["CapEff"], 16) or int(status["CapBnd"], 16) or int(status["NoNewPrivs"]) != 1:
        raise RuntimeError("Unexpected process permissions")
    if {name for _, name in socket.if_nameindex()} != {"lo"} or Path("/var/run/docker.sock").exists():
        raise RuntimeError("Unexpected network interface or Docker socket")
    for mount, readonly in [("/", True), ("/source", True), ("/output", False)]:
        if bool(os.statvfs(mount).f_flag & os.ST_RDONLY) != readonly:
            raise RuntimeError(f"Unexpected mount permissions on {mount}")
    cg = Path("/sys/fs/cgroup")
    for name, expected in [("memory.max", str(16 << 30)), ("memory.swap.max", "0"), ("pids.max", "64")]:
        if (cg / name).read_text().strip() != expected:
            raise RuntimeError(f"Unexpected cgroup setting: {name}")
    quota, period = map(int, (cg / "cpu.max").read_text().split())
    if quota != 2 * period:
        raise RuntimeError("Expected two CPU equivalents")


if __name__ == "__main__":
    if sys.argv[1:] == ["runtime"]:
        check_runtime()
    check_devices()
    if sys.argv[1:] != ["devices"]:
        check_packages()
    if sys.argv[1:] == ["runtime"]:
        print("Runtime checks passed: CPU build, no GPU devices or Docker socket, only loopback, "
              "UID/GID 1000:1000, zero capabilities, restricted cgroup, read-only root and source, writable output.")
