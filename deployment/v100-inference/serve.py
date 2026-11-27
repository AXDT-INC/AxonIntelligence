"""Private AxonAI HTTP API on a filesystem Unix socket."""
import argparse
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from functools import partial
from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path
from queue import Empty, SimpleQueue
import re
import signal
from socket import SHUT_RD
from socketserver import ThreadingMixIn, UnixStreamServer
import sys
import threading
from uuid import UUID

from infer import InvalidPrompt, load_resident, uuids

MAX_INPUT = 2048
MAX_OUTPUT = 4096
LEGACY_DEFAULT_OUTPUT = 256
WAITING = 2
MAX_BODY = 64 * 1024


@dataclass(eq=False)
class Generation:
    prompt: str
    limit: int
    request_id: str | None
    cancelled: threading.Event = field(default_factory=threading.Event)
    reply: Future = field(default_factory=Future)
    # At most one chunk per generated token plus the terminal event, bounded
    # by the admitted token cap. A slow socket must never stall the GPU worker.
    events: SimpleQueue = field(default_factory=SimpleQueue)

    def delta(self, text):
        self.events.put({"type": "delta", "text": text})

    def finish(self, status, payload):
        self.reply.set_result((status, payload))
        if self.request_id:
            event = dict(payload, type="done") if status == 200 else dict(payload, type="error", status=status)
            self.events.put(event)


class Worker(threading.Thread):
    def __init__(self, load, done, max_output=MAX_OUTPUT):
        super().__init__(name="resident-inference")
        self.load = load
        self.done = done
        self.max_output = max_output
        self.condition = threading.Condition()
        self.pending = deque()
        self.streams = {}
        self.state = "loading"

    def readiness(self):
        with self.condition:
            return (200 if self.state == "ready" else 503), {"status": self.state}

    def admit(self, prompt, limit, request_id=None):
        with self.condition:
            if self.state != "ready":
                return 503, {"error": "Service unavailable"}
            if request_id in self.streams:
                return 409, {"error": "Generation already in progress"}
            if len(self.pending) >= WAITING:
                return 429, {"error": "Inference queue full"}
            job = Generation(prompt, limit, request_id)
            if request_id:
                self.streams[request_id] = job
            self.pending.append(job)
            self.condition.notify_all()
            return 200, job

    def submit(self, prompt, limit):
        status, job = self.admit(prompt, limit)
        return job.reply.result() if status == 200 else (status, job)

    def cancel(self, request_id, expected=None):
        with self.condition:
            job = self.streams.get(request_id)
            if job is None or (expected is not None and job is not expected):
                return
            job.cancelled.set()
            if job in self.pending:
                self.pending.remove(job)
                del self.streams[request_id]
                job.finish(200, {"text": "", "input_tokens": 0, "output_tokens": 0, "finish_reason": "cancelled"})

    def stop(self, state="stopping"):
        with self.condition:
            if self.state != "failed":
                self.state = state
            for job in self.streams.values():
                job.cancelled.set()
            while self.pending:
                job = self.pending.popleft()
                self.streams.pop(job.request_id, None)
                job.finish(503, {"error": "Service unavailable"})
            self.condition.notify_all()

    def run(self):
        job = None
        try:
            resident = self.load()
            with self.condition:
                if self.state == "loading":
                    self.state = "ready"
                    self.condition.notify_all()
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.pending or self.state != "ready")
                    if self.state != "ready":
                        return
                    job = self.pending.popleft()
                try:
                    if job.request_id:
                        result = resident.generate(job.prompt, job.limit, on_text=job.delta, cancelled=job.cancelled)
                    else:
                        result = resident.generate(job.prompt, job.limit)
                    status = 200
                except InvalidPrompt as exc:
                    status, result = 400, {"error": str(exc)}
                with self.condition:
                    job.finish(status, result)
                    self.streams.pop(job.request_id, None)
                    self.condition.notify_all()
                del result
                job = None
        except Exception as exc:
            self.stop("failed")
            if job is not None:
                job.finish(500, {"error": "Inference failed"})
            detail = str(exc) if job is None else type(exc).__name__
            print(f"Resident worker failed: {detail}", file=sys.stderr, flush=True)
        finally:
            self.done.set()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # Keep prompts, URLs and responses out of access logs.

    def respond(self, status, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/ready":
            self.respond(*self.server.worker.readiness())
        else:
            self.respond(404, {"error": "Unknown endpoint"})

    def stream(self, job):
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            event = {"type": "start"}
            while True:
                self.wfile.write(json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n")
                self.wfile.flush()
                if event["type"] in ("done", "error"):
                    return
                try:
                    event = job.events.get(timeout=1)
                except Empty:
                    event = {"type": "ping"}
        finally:
            self.server.worker.cancel(job.request_id, expected=job)

    def do_POST(self):
        stop = re.fullmatch(r"/generations/([0-9a-f-]{36})/stop", self.path)
        if self.path not in ("/generate", "/generate/stream") and not stop:
            return self.respond(404, {"error": "Unknown endpoint"})
        if self.headers.get_content_type() != "application/json":
            return self.respond(415, {"error": "Expected application/json"})
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") is not None or len(lengths) != 1:
            return self.respond(400, {"error": "A single Content-Length is required"})
        length = lengths[0]
        if not length.isascii() or not length.isdecimal():
            return self.respond(400, {"error": "Invalid Content-Length"})
        if len(length) > 5 or int(length) > MAX_BODY:
            return self.respond(413, {"error": "Request body exceeds 64 KiB"})
        raw = self.rfile.read(int(length))
        try:
            if len(raw) != int(length):
                raise ValueError
            data = json.loads(raw.decode("utf-8"))
            if stop:
                request_id = stop[1]
                if str(UUID(request_id)) != request_id or data != {}:
                    raise ValueError
                self.server.worker.cancel(request_id)
                return self.respond(200, {"ok": True})
            streaming = self.path == "/generate/stream"
            allowed = {"prompt", "max_new_tokens", "request_id"} if streaming else {"prompt", "max_new_tokens"}
            if not isinstance(data, dict) or data.keys() - allowed:
                raise ValueError
            if streaming:
                request_id = data.get("request_id")
                if not isinstance(request_id, str) or str(UUID(request_id)) != request_id:
                    raise ValueError
            prompt = data.get("prompt")
            maximum = self.server.worker.max_output
            limit = data.get("max_new_tokens", maximum if streaming else min(maximum, LEGACY_DEFAULT_OUTPUT))
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError
            if type(limit) is not int or not 1 <= limit <= maximum:
                raise ValueError
            prompt.encode("utf-8")  # Reject unpaired JSON surrogate escapes before tokenization.
        except (ValueError, RecursionError):
            return self.respond(400, {"error": "Invalid generation request"})
        if streaming:
            status, result = self.server.worker.admit(prompt, limit, request_id)
            if status != 200:
                return self.respond(status, result)
            return self.stream(result)
        self.respond(*self.server.worker.submit(prompt, limit))


class Server(ThreadingMixIn, UnixStreamServer):
    def __init__(self, path, worker):
        self.worker = worker
        # Bound partial HTTP requests as well as the inference FIFO.
        self.connections = set()
        self.lock = threading.Lock()
        super().__init__(str(path), Handler)

    def process_request(self, request, address):
        request.settimeout(5)
        with self.lock:
            if len(self.connections) >= 12:
                self.shutdown_request(request)
                return
            self.connections.add(request)
        try:
            super().process_request(request, address)
        except BaseException:
            with self.lock:
                self.connections.discard(request)
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            with self.lock:
                self.connections.discard(request)

    def shutdown(self):
        super().shutdown()
        # Wake handlers reading incomplete requests. Active replies can still be written.
        with self.lock:
            for connection in self.connections:
                try:
                    connection.shutdown(SHUT_RD)
                except OSError:
                    pass  # The handler may have already closed its socket.

    def handle_error(self, request, address):
        if not isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            super().handle_error(request, address)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/model"))
    parser.add_argument("--reserved-uuids", type=uuids, required=True,
                        help="Two dedicated UUIDs already excluded from AxonOS scheduling, in CUDA order")
    parser.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT,
                        help=f"Stream default and request ceiling (1 through {MAX_OUTPUT})")
    args = parser.parse_args()
    if not 1 <= args.max_output_tokens <= MAX_OUTPUT:
        parser.error(f"--max-output-tokens must be between 1 and {MAX_OUTPUT}")
    if len(args.reserved_uuids) != 2:
        parser.error("Exactly two reserved GPU UUIDs are required")

    done = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: done.set())
    worker = Worker(partial(load_resident, args.model, args.reserved_uuids, MAX_INPUT, args.max_output_tokens),
                    done, args.max_output_tokens)
    old_mask = os.umask(0o177)
    try:
        server = Server(args.socket, worker)
    finally:
        os.umask(old_mask)
    with server:
        try:
            listener = threading.Thread(target=server.serve_forever)
            listener.start()
            worker.start()
            done.wait()
        finally:
            worker.stop()
            server.shutdown()
            listener.join()
            worker.join()
            args.socket.unlink()
    return int(worker.state == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
