"""Measure a private V100 service at its advertised output cap using a synthetic prompt."""
import argparse
from contextlib import closing
import hashlib
import http.client
import json
from pathlib import Path
import socket
import time
from uuid import uuid4


def full_input(model):
    from infer import SYSTEM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    prefix = 'Synthetic counter calibration. Ignore the following padding marks:'
    task = ('\nList the positive integers in order from 1 to 20000, separated by commas. '
            'Write every number literally, starting at 1. Do not abbreviate, summarize, '
            'use ellipses, or provide code. Keep counting until the whole list is complete.')

    def size(prompt):
        chat = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return len(tokenizer.encode(text, add_special_tokens=False))

    prompt = prefix + ' x' * (2048 - size(prefix + task)) + task
    if size(prompt) != 2048:
        raise RuntimeError('Fixture does not reach exactly 2,048 rendered tokens with this tokenizer')
    return prompt


def connect(path):
    conn = http.client.HTTPConnection('localhost', timeout=30)
    conn.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.sock.settimeout(30)
    conn.sock.connect(str(path))
    return conn


def check(path, prompt, cap, stop_after=None):
    request_id = str(uuid4())
    body = {'prompt': prompt, 'max_new_tokens': cap, 'request_id': request_id}
    start = time.monotonic()
    first = stop_time = None
    chunks = []
    with closing(connect(path)) as conn:
        conn.request('POST', '/generate/stream', json.dumps(body), {'Content-Type': 'application/json'})
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f'Stream admission returned {response.status}')
        while line := response.readline(262145):
            event = json.loads(line)
            kind = event['type']
            if kind == 'delta':
                if not event['text']:
                    raise RuntimeError('Empty text delta')
                if first is None:
                    first = time.monotonic()
                chunks.append(event['text'])
                if stop_after and len(chunks) == stop_after:
                    stop_time = time.monotonic()
                    with closing(connect(path)) as control:
                        control.request('POST', f'/generations/{request_id}/stop', '{}',
                                        {'Content-Type': 'application/json'})
                        reply = control.getresponse()
                        if reply.status != 200:
                            raise RuntimeError(f'Stop returned {reply.status}')
                        reply.read()
            elif kind == 'done':
                end = time.monotonic()
                if ''.join(chunks) != event['text']:
                    raise RuntimeError('Published deltas differ from the completed text')
                if first is None:
                    raise RuntimeError('No visible streamed text before completion')
                return {'input_tokens': event['input_tokens'], 'output_tokens': event['output_tokens'],
                        'finish_reason': event['finish_reason'], 'text_deltas': len(chunks),
                        'first_text_s': round(first - start, 3), 'elapsed_s': round(end - start, 3),
                        'stop_to_done_s': round(end - stop_time, 3) if stop_time is not None else None,
                        'text_sha256': hashlib.sha256(event['text'].encode()).hexdigest()}
            elif kind not in ('start', 'ping'):
                raise RuntimeError(f'Unexpected stream event: {kind}')
    raise RuntimeError('Stream closed without a final event')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('/model'))
    parser.add_argument('--cap', type=int, required=True)
    args = parser.parse_args()
    prompt = full_input(args.model)
    stopped = check(args.socket, prompt, args.cap, stop_after=3)
    if stopped['finish_reason'] != 'cancelled' or stopped['output_tokens'] >= args.cap:
        raise RuntimeError('Worker did not stop the long prompt early')
    print(json.dumps({'cancel': stopped}), flush=True)
    result = check(args.socket, prompt, args.cap)
    print(json.dumps({'full': result}), flush=True)
    if (result['input_tokens'] != 2048 or result['output_tokens'] != args.cap
            or result['finish_reason'] != 'length' or result['text_deltas'] <= 1):
        raise RuntimeError('Long request did not stream to the configured token boundary')
    recovery = check(args.socket, 'Reply with just the single word OK.', 64)
    print(json.dumps({'recovery': recovery}), flush=True)
    if recovery['finish_reason'] != 'stop' or recovery['output_tokens'] >= 64:
        raise RuntimeError('Follow-up request did not recover and reach EOS')


if __name__ == '__main__':
    main()
