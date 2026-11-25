"""Private AxonAI HTTP API on a filesystem Unix socket."""
import argparse
from collections import deque
from concurrent.futures import Future
from functools import partial
from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path
import signal
from socket import SHUT_RD
from socketserver import ThreadingMixIn, UnixStreamServer
import sys
import threading

from infer import InvalidPrompt, load_resident, uuids

MAX_INPUT = 2048
MAX_OUTPUT = 256
WAITING = 2
MAX_BODY = 64 * 1024


class Worker(threading.Thread):
    def __init__(self, load, done):
        super().__init__(name="resident-inference")
        self.load = load
        self.done = done
        self.condition = threading.Condition()
        self.pending = deque()
        self.state = "loading"

    def readiness(self):
        with self.condition:
            return (200 if self.state == "ready" else 503), {"status": self.state}

    def submit(self, prompt, limit):
        with self.condition:
            if self.state != "ready":
                return 503, {"error": "Service unavailable"}
            if len(self.pending) >= WAITING:
                return 429, {"error": "Inference queue full"}
            reply = Future()
            self.pending.append((prompt, limit, reply))
            self.condition.notify()
        return reply.result()

    def stop(self, state="stopping"):
        with self.condition:
            if self.state != "failed":
                self.state = state
            while self.pending:
                _, _, reply = self.pending.popleft()
                reply.set_result((503, {"error": "Service unavailable"}))
            self.condition.notify_all()

    def run(self):
        reply = None
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
                    prompt, limit, reply = self.pending.popleft()
                try:
                    result = 200, resident.generate(prompt, limit)
                except InvalidPrompt as exc:
                    result = 400, {"error": str(exc)}
                reply.set_result(result)
                del prompt, result
                reply = None
        except Exception as exc:
            self.stop("failed")
            if reply is not None:
                reply.set_result((500, {"error": "Inference failed"}))
            detail = str(exc) if reply is None else type(exc).__name__
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

    def do_POST(self):
        if self.path != "/generate":
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
            if not isinstance(data, dict) or data.keys() - {"prompt", "max_new_tokens"}:
                raise ValueError
            prompt = data.get("prompt")
            limit = data.get("max_new_tokens", MAX_OUTPUT)
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError
            if type(limit) is not int or not 1 <= limit <= MAX_OUTPUT:
                raise ValueError
            prompt.encode("utf-8")  # Reject unpaired JSON surrogate escapes before tokenization.
        except (ValueError, RecursionError):
            return self.respond(400, {"error": "Expected a nonempty prompt and optional integer max_new_tokens (1-256)"})
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
    args = parser.parse_args()
    if len(args.reserved_uuids) != 2:
        parser.error("Exactly two reserved GPU UUIDs are required")

    done = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: done.set())
    worker = Worker(partial(load_resident, args.model, args.reserved_uuids, MAX_INPUT, MAX_OUTPUT), done)
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
