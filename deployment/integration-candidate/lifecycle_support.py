"""Narrow ownership helpers; no numerical or publication operations."""
from contextlib import contextmanager
import signal

@contextmanager
def acquisition_signals():
    # Defer SIGINT across resource-return/registration instructions. Restoration
    # remains inside the owner's outer BaseException guard. Not SIGKILL safety.
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)

class OwnedDescriptors:
    """Track authentication descriptors before any operation that may fail."""
    def __init__(self, diagnostic):
        self.fds = {}
        self.diagnostic = diagnostic

    def acquire(self, operation):
        import os
        with acquisition_signals():
            fd = operation()
            self.fds[fd] = None
            info = os.fstat(fd)
            self.fds[fd] = (info.st_dev, info.st_ino)
        return fd

    def close(self, fd, strict=False):
        import os, errno
        if fd not in self.fds:
            return True
        try:
            info = os.fstat(fd)
            identity = self.fds[fd]
            if identity is not None and identity != (info.st_dev, info.st_ino):
                raise RuntimeError('descriptor reused; refusing to close unrelated resource')
            with acquisition_signals():
                os.close(fd)
                self.fds.pop(fd, None)
        except BaseException as error:
            self.diagnostic('descriptor_close', error)
            try:
                os.fstat(fd)
            except OSError as probe:
                if probe.errno == errno.EBADF:
                    self.fds.pop(fd, None)
            except BaseException as probe:
                self.diagnostic('descriptor_status', probe)
            if strict:
                raise
        return fd not in self.fds

    def close_all(self):
        for fd in list(self.fds):
            self.close(fd)

    def pipe(self):
        import os
        with acquisition_signals():
            pair = os.pipe2(os.O_CLOEXEC)
            for fd in pair:
                self.fds[fd] = None
            for fd in pair:
                info = os.fstat(fd)
                self.fds[fd] = (info.st_dev, info.st_ino)
        return pair
