"""Supervisor-only authenticated, write-once local ledger with transactional attempt claiming."""
import fcntl
import hmac
import json
import os
from pathlib import Path
import sqlite3
import stat

from runner import require


def canonical(v):
    return json.dumps(v, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()


def private(path, directory=False):
    info = path.lstat()
    kind = stat.S_ISDIR if directory else stat.S_ISREG
    require(kind(info.st_mode) and info.st_uid == os.getuid() and info.st_mode & 0o077 == 0
            and (directory or info.st_nlink == 1), "invalid private path")
    require(not any((p / ".git" / "HEAD").is_file() or (p / ".git").is_file()
                    for p in (path.resolve(), *path.resolve().parents)),
            "private data must be outside Git")


class Ledger:
    def __init__(self, root, keyfile, session, *, readonly=False):
        root, keyfile = Path(root), Path(keyfile)
        private(root, True)
        private(keyfile)
        self.key = keyfile.read_bytes()
        require(len(self.key) == 32, "expected a private 256-bit signing key")
        self.session = session
        self.fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            lock = fcntl.LOCK_SH if readonly else fcntl.LOCK_EX
            fcntl.flock(self.fd, lock | fcntl.LOCK_NB)
            db = root / "receipts.sqlite"
            if not readonly and not db.exists():
                os.close(os.open(db, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
            private(db)
            if readonly:
                self.db = sqlite3.connect(db.resolve().as_uri()+"?mode=ro", uri=True)
                require(self.get("session") == session, "unauthenticated session")
                return
            self.db = sqlite3.connect(db)
            self.db.execute("PRAGMA synchronous=FULL")
            with self.db:
                self.db.execute("CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, payload BLOB NOT NULL, mac TEXT NOT NULL)")
            os.fsync(self.fd)
            self.bind("session", session)
        except BaseException:
            if hasattr(self, "db"):
                self.db.close()
            os.close(self.fd)
            raise

    def mac(self, name, raw):
        return hmac.digest(self.key, canonical([self.session, name]) + b"\0" + raw, "sha256").hex()

    def get(self, name):
        record = self.db.execute("SELECT payload,mac FROM events WHERE id=?", (name,)).fetchone()
        if record is None:
            return None
        raw, signature = record
        require(hmac.compare_digest(self.mac(name, raw), signature), "receipt authentication failed")
        return json.loads(raw)

    def put(self, name, value):
        self.put_many({name: value})

    def put_many(self, events):
        with self.db:
            for name, value in events.items():
                require(value is not None, "null ledger entry")
                raw = canonical(value)
                self.db.execute("INSERT INTO events VALUES(?,?,?)", (name, raw, self.mac(name, raw)))

    def bind(self, name, value):
        saved = self.get(name)
        if saved is None:
            self.put(name, value)
        else:
            require(saved == value, "attempt binding changed")

    def close(self):
        self.db.close()
        os.close(self.fd)
