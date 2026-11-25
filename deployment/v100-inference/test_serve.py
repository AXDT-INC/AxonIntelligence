"""CPU-only checks of the serving boundary with an in-memory inference stand-in."""
import http.client
import json
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

from infer import InvalidPrompt, Resident
from serve import MAX_INPUT, MAX_OUTPUT, Worker


class Serving(unittest.TestCase):
    def test_rendered_budget(self):
        tokenizer = Mock()
        resident = Resident(None, tokenizer, [], set(), MAX_INPUT, MAX_OUTPUT)
        for count in (2048, 2049):
            tokenizer.return_value.input_ids = SimpleNamespace(shape=(1, count))
            if count == 2048:
                self.assertEqual(resident.encode("hello").shape, (1, 2048))
            else:
                with self.assertRaises(InvalidPrompt):
                    resident.generate("hello", 1)
        messages, = tokenizer.apply_chat_template.call_args.args
        self.assertEqual([m["role"] for m in messages], ["system", "user"])

    def test_fifo_and_stop(self):
        started = threading.Event()
        release = threading.Event()
        order = []

        def generate(prompt, limit):
            order.append(prompt)
            started.set()
            if not release.wait(5):
                raise AssertionError("timed out")
            return prompt

        load = Mock(return_value=SimpleNamespace(generate=generate))
        worker = Worker(load, threading.Event())
        self.assertEqual(worker.readiness()[0], 503)
        worker.start()
        with ThreadPoolExecutor(3) as callers:
            try:
                with worker.condition:
                    self.assertTrue(worker.condition.wait_for(lambda: worker.state == "ready", timeout=3))
                for stopping in (False, True):
                    release.clear()
                    started.clear()
                    first = callers.submit(worker.submit, "one", 1)
                    self.assertTrue(started.wait(3))
                    waiting = []
                    for name in ("two", "three"):
                        waiting.append(callers.submit(worker.submit, name, 1))
                        with worker.condition:
                            self.assertTrue(worker.condition.wait_for(
                                lambda: len(worker.pending) == len(waiting), timeout=3))
                    self.assertEqual(worker.readiness()[0], 200)
                    self.assertEqual(worker.submit("overflow", 1)[0], 429)
                    if stopping:
                        worker.stop()
                        for job in waiting:
                            self.assertEqual(job.result(3)[0], 503)
                    release.set()
                    self.assertEqual(first.result(3), (200, "one"))
                    if not stopping:
                        self.assertEqual([j.result(3) for j in waiting], [(200, "two"), (200, "three")])
            finally:
                worker.stop()
                release.set()
                worker.join(5)
        self.assertEqual(order, ["one", "two", "three", "one"])
        load.assert_called_once()
        self.assertFalse(worker.is_alive())

    def test_private_http_process(self):
        # Run the actual entrypoint, replacing only the GPU loader in the child.
        child = """
import serve
class Resident:
    def generate(self, prompt, limit):
        if prompt == "long":
            raise serve.InvalidPrompt("too many tokens")
        return {"text": prompt, "limit": limit}
serve.load_resident = lambda *args: Resident()
raise SystemExit(serve.main())
"""
        pair = ",".join(f"GPU-00000000-0000-0000-0000-{i:012d}" for i in (1, 2))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "inference.sock")
            proc = subprocess.Popen([sys.executable, "-B", "-c", child, "--reserved-uuids", pair,
                                     "--socket", str(path)], cwd=Path(__file__).parent)
            try:
                def request(pathname, payload=None):
                    with socket.socket(socket.AF_UNIX) as sock:
                        sock.settimeout(3)
                        sock.connect(str(path))
                        conn = http.client.HTTPConnection("localhost")
                        conn.sock = sock
                        conn.request("GET" if payload is None else "POST", pathname, body=payload,
                                     headers={"Content-Type": "application/json"})
                        res = conn.getresponse()
                        return res.status, json.loads(res.read())

                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and proc.poll() is None:
                    if path.exists() and request("/ready")[0] == 200:
                        break
                    time.sleep(0.01)
                self.assertEqual(request("/ready"), (200, {"status": "ready"}))
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                for payload in (b"[]", b"{}", b"{", b'{"prompt":" "}', b'{"prompt":"a","history":[]}',
                                b'{"prompt":"long"}', b'{"prompt":"a","max_new_tokens":true}',
                                b'{"prompt":"a","max_new_tokens":257}', b'{"prompt":"a","max_new_tokens":0}'):
                    self.assertEqual(request("/generate", payload)[0], 400)
                self.assertEqual(request("/generate", b" " * 65537)[0], 413)
                self.assertEqual(request("/generate", b'{"prompt":"first","max_new_tokens":1}'),
                                 (200, {"text": "first", "limit": 1}))
                self.assertEqual(request("/generate", b'{"prompt":"second"}'),
                                 (200, {"text": "second", "limit": 256}))
                with socket.socket(socket.AF_UNIX) as incomplete:
                    incomplete.connect(str(path))
                    incomplete.sendall(b"POST /generate HTTP/1.1\r\n")
                    self.assertEqual(request("/ready")[0], 200)
                    proc.send_signal(signal.SIGTERM)
                    self.assertEqual(proc.wait(3), 0)
                self.assertFalse(path.exists())
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
