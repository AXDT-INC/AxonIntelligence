"""Apply a network restriction to this Python invocation and its children."""
import ctypes as c
import errno
import os
import socket
import sys

class Comparison(c.Structure):
    _fields_ = [('arg', c.c_uint), ('op', c.c_uint), ('a', c.c_uint64), ('b', c.c_uint64)]

lib = c.CDLL("libseccomp.so.2")
lib.seccomp_init.argtypes = [c.c_uint32]
lib.seccomp_init.restype = c.c_void_p
lib.seccomp_rule_add_array.argtypes = [c.c_void_p, c.c_uint32, c.c_int, c.c_uint, c.POINTER(Comparison)]
lib.seccomp_load.argtypes = [c.c_void_p]
lib.seccomp_syscall_resolve_name.argtypes = [c.c_char_p]
ctx = lib.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW
if not ctx:
    raise RuntimeError("seccomp_init")
for name in (b"socket", b"socketpair", b"io_uring_setup"):
    number = lib.seccomp_syscall_resolve_name(name)
    if number < 0:
        raise RuntimeError(name)
    rule = None if name == b"io_uring_setup" else Comparison(0, 1, socket.AF_UNIX, 0)  # SCMP_CMP_NE
    rc = lib.seccomp_rule_add_array(ctx, 0x00050000 | errno.EPERM, number, int(rule is not None),
                                    c.byref(rule) if rule is not None else None)
    if rc != 0:
        raise RuntimeError((name,rc))
if lib.seccomp_load(ctx) != 0:
    raise RuntimeError("seccomp_load")
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
except PermissionError:
    pass
else:
    sock.close()
    raise RuntimeError("Internet socket available")
os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
