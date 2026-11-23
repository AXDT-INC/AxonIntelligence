import ctypes
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/data/axonai')
BASE = ROOT / 'models/base/Qwen3.8-27B'
ADAPTER = ROOT / 'engineering-maintenance/main-sft-run-20261029T150039Z/final_adapter'
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
ADAPTER_SHA = '5280cf5f30265038a6fee6aa57955b829d387b7d78fb9fe6afab86d65ac50978'

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def binding(path):
    return {'path': str(path), 'sha256': sha(path), 'bytes': Path(path).stat().st_size}

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n'); f.flush(); os.fsync(f.fileno())

def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, sort_keys=True) + '\n'); f.flush(); os.fsync(f.fileno())

def verify(path, expected):
    result = binding(path)
    if result['sha256'] != expected:
        raise ValueError(f'Artifact hash mismatch: {path}')
    return result

def restrict_writes(output):
    """Landlock restricts this process and descendants: only output and /dev writable."""
    libc = ctypes.CDLL(None, use_errno=True)
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 3:
        raise RuntimeError('Landlock ABI >=3 required; refusing unconfined GPU worker')
    # WRITE_FILE, REMOVE_DIR/FILE, MAKE_CHAR/DIR/REG/SOCK/FIFO/BLOCK/SYM, REFER, TRUNCATE.
    rights = (1 << 1) | sum(1 << bit for bit in range(4, 15))
    class Ruleset(ctypes.Structure):
        _fields_ = [('handled_access_fs', ctypes.c_uint64)]
    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]
    attr = Ruleset(rights)
    fd = libc.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'landlock_create_ruleset')
    try:
        for path in [Path(output).resolve(), Path('/dev'), Path('/proc/self/task')]:
            pfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(((1 << 1) | (1 << 14)) if str(path).startswith('/proc/') else rights, pfd)
                if libc.syscall(445, fd, 1, ctypes.byref(rule), 0) != 0:
                    raise OSError(ctypes.get_errno(), 'landlock_add_rule')
            finally:
                os.close(pfd)
        if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.syscall(446, fd, 0) != 0:
            raise OSError(ctypes.get_errno(), 'landlock_restrict_self')
    finally:
        os.close(fd)
    return {'landlock_abi': abi, 'writable_paths': [str(Path(output).resolve()), '/dev'],
            'original_artifacts_writable': False, 'inherited_by_descendants': True, 'own_thread_metadata_write': '/proc/self/task (WRITE_FILE/TRUNCATE only)'}
