"""Private, bounded JSON-lines worker. Use only inside the isolated model container."""
import json
import sys
import signal
import resource
import time
from dataclasses import asdict

import runner as r
from live_backend import Backend, VOCAB, check_options, conditional_scores

MAX_LINE = 2**20
OPERATIONS = ("generate", "options", "probe")


def pairs(values):
    obj = dict(values)
    r.require(len(obj) == len(values), "duplicate field")
    return obj


def parse(raw):
    r.require(0 < len(raw) <= MAX_LINE, "invalid frame size")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)


def reject(_):
    raise r.IntegrityError("nonfinite JSON")


def fields(obj, names):
    r.require(type(obj) is dict and obj.keys() == set(names.split()), "unexpected private input fields")


def unpack(job):
    fields(job, "item_id rendered_sha256 input_ids options")
    r.require(type(job["input_ids"]) is list and len(job["input_ids"]) <= 65536, "invalid input budget")
    item = r.Item(job["item_id"], job["rendered_sha256"], tuple(job["input_ids"]))
    item.validate()
    r.require(max(item.input_ids) < VOCAB, "invalid input vocabulary")
    check_options(job["options"])
    return item


def fixtures(tokenizer):
    """Render two disposable tasks. The second fills the exact common input envelope."""
    text = "Return one short word describing a blue square."
    short = r.render_item("1"*32, text, tokenizer)
    size = r.CONTEXT - r.NEW_TOKENS
    pad = size - len(short.input_ids)
    r.require(pad > 0, "synthetic fixture exceeds budget")
    long = r.render_item("2"*32, text + " x"*pad, tokenizer)
    if len(long.input_ids) != size:
        # Find a neutral rendered prompt with exact length under the verified tokenizer.
        for filler in (" x", " q", " 7"):
            low, high = 1, 2*size
            while low <= high:
                count = (low + high)//2
                long = r.render_item("2"*32, text + filler*count, tokenizer)
                if len(long.input_ids) == size:
                    break
                if len(long.input_ids) < size:
                    low = count+1
                else:
                    high = count-1
            if len(long.input_ids) == size:
                break
    r.require(len(long.input_ids) == size, "unexpected synthetic pad tokenization")
    options = [tokenizer(s, add_special_tokens=False)["input_ids"] for s in (" blue", " square corner")]
    check_options(options)
    return [dict(asdict(i), input_ids=list(i.input_ids), options=options) for i in (short, long)]


def failure(backend_error):
    def raise_failure(*_):
        raise r.ItemFailure(backend_error)
    return raise_failure


def execute(backend, op, run_id, slot, job):
    item = unpack(job)
    r.require(op in OPERATIONS and r.opaque(run_id) and r.opaque(slot), "invalid assignment")
    observed = backend.runtime
    def decode(ids, limit):
        start = time.perf_counter()
        try:
            return backend.decode(ids, limit)
        except backend.torch.cuda.OutOfMemoryError:
            cuda = backend.torch.cuda
            metrics = r.Measurement(time.perf_counter()-start, None,
                tuple(cuda.max_memory_allocated(i) for i in range(2)),
                tuple(cuda.max_memory_reserved(i) for i in range(2)),
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            raise r.ItemFailure("oom", metrics) from None
        except RuntimeError:
            raise r.ItemFailure("backend_error") from None
    if op == "generate":
        row = r.run_item(run_id, slot, item, observed, observed, decode)
        m = row["measurement"]
        count = row["output_tokens"]
        rate = count / m["latency_s"] if m and count else None
        decode_s = m["latency_s"] - m["first_token_s"] if count and m else 0
        decode_rate = (count-1)/decode_s if count and count > 1 and decode_s > 0 else None
        return {"receipt": row, "tokens_per_second": rate, "decode_tokens_per_second": decode_rate}
    out = {"op": op, "input_sha256": r.digest(item.input_ids), "options_sha256": r.digest(job["options"]),
           "status": "ok", "value": None}
    backend.start_aux()
    start = time.monotonic()
    try:
        if len(item.input_ids) + r.NEW_TOKENS > r.CONTEXT:
            out["status"] = "context_overflow"
        elif op == "options":
            out["value"] = conditional_scores(item.input_ids, job["options"], backend.option_logprobs)
        else:
            out["value"] = backend.probe(item.input_ids)
    except backend.torch.cuda.OutOfMemoryError:
        out["status"] = "oom"
    except RuntimeError:
        out["status"] = "backend_error"
    except r.ItemFailure as exc:
        out["status"] = exc.status
    out["peaks"] = backend.aux_peaks()
    out["latency_s"] = time.monotonic() - start
    return out


def emit(value):
    print(json.dumps(value, allow_nan=False, separators=(",", ":")), flush=True)


def main():
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(930)
    line = sys.stdin.buffer.readline(MAX_LINE + 1)
    request = parse(line)
    fields(request, "mode image uuids common run_id slot_ticket synthetic")
    mode = request["mode"]
    r.require(type(request["synthetic"]) is bool and r.opaque(request["run_id"]) and
              r.opaque(request["slot_ticket"]), "invalid session")
    r.require(request["synthetic"] or mode in {"bare", "adapter"}, "control mode requires synthetic fixtures")
    backend = Backend(mode, request["image"], request["uuids"], request["common"])
    demo = fixtures(backend.tokenizer) if request["synthetic"] else []
    if mode != "discover":
        backend.identity["warmup"] = backend.warmup()
    emit({"ready": backend.identity, "fixtures": demo})
    if mode == "discover":
        return
    while True:
        signal.alarm(120)
        line = sys.stdin.buffer.readline(MAX_LINE + 1)
        if not line:
            break
        signal.alarm(190)
        task = parse(line)
        fields(task, "op job")
        if request["synthetic"]:
            r.require(task["job"] in demo, "unregistered synthetic input")
        else:
            r.require(task["op"] != "probe", "probe requires synthetic control")
        result = execute(backend, task["op"], request["run_id"], request["slot_ticket"], task["job"])
        emit({"request_sha256": r.digest(task), "result": result})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Report only fixed classes; exception strings can contain private paths or model text.
        if isinstance(error, r.IntegrityError):
            category = "identity_or_contract"
        elif isinstance(error, ImportError):
            category = "missing_dependency"
        elif isinstance(error, (OSError, TimeoutError)):
            category = "host_io"
        else:
            category = "model_runtime"
        emit({"fatal": category})
        sys.exit(2)
