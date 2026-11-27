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
from unittest.mock import Mock, patch

from infer import InvalidPrompt, Resident
from serve import MAX_INPUT, MAX_OUTPUT, Server, Worker


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
        card = Path(__file__).resolve().parents[1] / "public-release/hub/bf16/README.md"
        section = card.read_text().split("## Recommended system prompt", 1)[1]
        published = section.split("```text\n", 1)[1].split("\n```", 1)[0]
        self.assertEqual(messages, [{"role": "system", "content": published}, {"role": "user", "content": "hello"}])
        for marker in ("<|im_start|>", "<|im_end|>"):
            with self.assertRaises(InvalidPrompt):
                resident.encode(marker)

    def test_utf8_stream_and_terminal_reason(self):
        # The pinned inference image includes the native incremental decoder.
        try:
            from tokenizers import Tokenizer, decoders, models
        except ImportError:
            self.skipTest("CPU-only stdlib installation lacks the inference tokenizer")
        backend = Tokenizer(models.WordLevel(vocab={"<0xE2>": 0, "<0x82>": 1, "<0xAC>": 2,
                                                    "x": 3, "<end>": 4}))
        backend.decoder = decoders.ByteFallback()
        backend.add_special_tokens(["<end>"])
        resident = Resident(None, SimpleNamespace(backend_tokenizer=backend), [], {4}, MAX_INPUT, MAX_OUTPUT)
        for tokens, limit, reason in (([3, 0, 1, 2, 4], 5, "stop"), ([3, 0, 1, 2], 4, "length"),
                                      ([3, 0], 4, "cancelled"), ([], 4, "cancelled")):
            def decode(ids, actual_limit, *, on_token, cancelled):
                for token in tokens:
                    on_token(token)
                return backend.decode(tokens), tokens, 0, 0

            chunks = []
            with patch.object(resident, "encode", return_value=SimpleNamespace(shape=(1, 5))), \
                 patch.object(resident, "decode", side_effect=decode):
                reply = resident.generate("hi", limit, on_text=chunks.append)
            self.assertEqual(reply["text"], "".join(chunks))
            self.assertEqual(reply["finish_reason"], reason)
            if 2 in tokens:
                self.assertEqual(chunks, ["x", "€"])

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

    def test_streaming_stop_and_queue_over_unix_socket(self):
        first_chunk = threading.Event()
        calls = []

        class FakeResident:
            def generate(self, prompt, limit, *, on_text, cancelled):
                calls.append(prompt)
                if prompt == "invalid":
                    raise InvalidPrompt("Rendered prompt exceeds the limit")
                on_text("α")
                if prompt == "hold":
                    first_chunk.set()
                    if not cancelled.wait(5):
                        raise RuntimeError("stream did not stop")
                    return dict(text="α", input_tokens=4, output_tokens=1, finish_reason="cancelled")
                on_text("β")
                return dict(text="αβ", input_tokens=4, output_tokens=2, finish_reason="length")

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp, "s.sock"))
            worker = Worker(FakeResident, threading.Event(), max_output=1024)
            with Server(path, worker) as server:
                listener = threading.Thread(target=server.serve_forever)
                listener.start()
                worker.start()
                with worker.condition:
                    self.assertTrue(worker.condition.wait_for(lambda: worker.state == "ready", timeout=3))

                def request(route, data):
                    conn = http.client.HTTPConnection("localhost")
                    conn.sock = socket.socket(socket.AF_UNIX)
                    conn.sock.settimeout(5)
                    conn.sock.connect(path)
                    conn.request("POST", route, body=json.dumps(data), headers={"Content-Type": "application/json"})
                    response = conn.getresponse()
                    self.addCleanup(response.close)
                    return response

                def stream(number, prompt, limit=1024):
                    return request("/generate/stream", {"prompt": prompt, "max_new_tokens": limit,
                                                        "request_id": f"00000000-0000-4000-a000-{number:012d}"})

                def stop(number):
                    return request(f"/generations/00000000-0000-4000-a000-{number:012d}/stop", {}).status

                def event(response):
                    while True:
                        data = json.loads(response.readline())
                        if data["type"] != "ping":
                            return data

                try:
                    running = stream(1, "hold")
                    self.assertEqual(event(running)["type"], "start")
                    self.assertTrue(first_chunk.wait(3))
                    self.assertEqual(event(running), {"type": "delta", "text": "α"})
                    self.assertEqual(stream(1, "duplicate").status, 409)
                    waiting = stream(2, "queued")
                    self.assertEqual(event(waiting)["type"], "start")
                    self.assertEqual(stop(2), 200)
                    self.assertEqual(event(waiting)["output_tokens"], 0)
                    self.assertEqual(stop(1), 200)
                    self.assertEqual(event(running), dict(type="done", text="α", input_tokens=4,
                                                         output_tokens=1, finish_reason="cancelled"))
                    self.assertEqual(stop(1), 200)
                    finished = stream(3, "finish", 2)
                    events = [event(finished) for _ in range(4)]
                    self.assertEqual([e["type"] for e in events], ["start", "delta", "delta", "done"])
                    self.assertEqual(events[-1]["finish_reason"], "length")
                    self.assertEqual("".join(e["text"] for e in events[1:3]), events[-1]["text"])
                    invalid = stream(4, "invalid")
                    self.assertEqual(event(invalid)["type"], "start")
                    self.assertEqual(event(invalid)["status"], 400)
                    self.assertEqual(stream(5, "over-limit", 1025).status, 400)
                    self.assertEqual(calls, ["hold", "finish", "invalid"])
                    # Closing the reader also cancels an otherwise quiet active turn.
                    abandoned = stream(6, "hold")
                    self.assertEqual(event(abandoned)["type"], "start")
                    self.assertEqual(event(abandoned)["type"], "delta")
                    abandoned.close()
                    with worker.condition:
                        self.assertTrue(worker.condition.wait_for(lambda: not worker.streams, timeout=4))
                    self.assertEqual(worker.readiness()[0], 200)
                finally:
                    worker.stop()
                    server.shutdown()
                    listener.join()
                    worker.join()

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
                                json.dumps({"prompt":"a","max_new_tokens":MAX_OUTPUT + 1}).encode(), b'{"prompt":"a","max_new_tokens":0}'):
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
