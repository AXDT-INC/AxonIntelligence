"""New replay harness, not historical source. Linux x86_64 Landlock ABI >= 4."""
import ctypes
import json
import os
import platform
import resource
import sys
from pathlib import Path

def confine(paths):
    assert platform.machine() == 'x86_64'
    libc = ctypes.CDLL(None, use_errno=True)
    def call(number, *args):
        result = libc.syscall(number, *args)
        if result < 0:
            raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
        return result
    abi = call(444, 0, 0, 1)
    assert abi >= 4, 'Landlock ABI >= 4 required'
    class Ruleset(ctypes.Structure):
        _fields_ = [('fs', ctypes.c_uint64), ('net', ctypes.c_uint64)]
    class Beneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('access', ctypes.c_uint64), ('fd', ctypes.c_int32)]
    # Deny all write/create/remove/rename/truncate operations except in paths.
    writes = sum(1 << i for i in [1,4,5,6,7,8,9,10,11,12,13,14])
    rule = Ruleset(writes, 3)  # Deny TCP connect/bind everywhere.
    fd = call(444, ctypes.byref(rule), ctypes.sizeof(rule), 0)
    try:
        for path in paths:
            p = Path(path)
            assert p.is_dir() and not p.is_symlink() and p.resolve() == p
            d = os.open(p, os.O_PATH | os.O_CLOEXEC)
            try:
                attr = Beneath(writes, d)
                call(445, fd, 1, ctypes.byref(attr), 0)
            finally:
                os.close(d)
        if libc.prctl(38, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'PR_SET_NO_NEW_PRIVS failed')
        call(446, fd, 0)
    finally:
        os.close(fd)
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return abi

if __name__ == '__main__':
    split = sys.argv.index('--')
    paths = sys.argv[1:split]
    command = sys.argv[split+1:]
    assert len(paths) == 2 and command
    confine(paths)
    # No inherited regular-file descriptors; only stdin/stdout/stderr survive.
    os.closerange(3, 65536)
    os.execv(command[0], command)
