#!/usr/bin/env python3
"""Independent bit-arithmetic numerical verifier; no converter import or GPU access."""
import argparse
import math
import os
import struct
import sys
from audit_io import (AuditError, CHUNK, CONVERTED, COPY_FILES, Directory, OUTPUT_MANIFEST,
                      SOURCE, SOURCE_SHA, canonical, checked_manifest, demand, emit_receipt,
                      model_metadata, parse_json, sha_bytes, validate_source, ReceiptDestination, LifecycleError)

def expected_half_bits(bits):
    """Integer BF16->binary16 reference. No float cast or converter helper.

    BF16 subnormals are below the half-minimum rounding threshold. For normal
    BF16 values below the half normal range, round the 8-bit significand using
    quotient/remainder and an explicit ties-to-even rule.
    """
    sign = bits & 0x8000
    exponent = (bits >> 7) & 255
    fraction = bits & 127
    demand(exponent != 255, 'nonfinite BF16 input')
    demand((bits & 0x7fff) <= 0x477f, 'FP16 overflow')  # largest BF16 <= 65504 = 65280
    if exponent == 0:
        return sign
    if exponent >= 113:
        return sign | ((exponent-112) << 10) | (fraction << 3)
    significand = 128+fraction
    if exponent >= 110:
        return sign | (significand << (exponent-110))
    shift = 110-exponent
    quotient, remainder = divmod(significand, 1 << shift)
    halfway = 1 << (shift-1)
    if remainder > halfway or (remainder == halfway and quotient & 1):
        quotient += 1
    return sign | quotient

def reference_value(bits):
    exponent = (bits >> 7) & 255
    fraction = bits & 127
    magnitude = math.ldexp(fraction, -133) if exponent == 0 else math.ldexp(128+fraction, exponent-134)
    return -magnitude if bits & 0x8000 else magnitude

def initial_statistics():
    return {'elements': 0, 'changed': 0, 'zeroed_nonzero': 0, 'output_subnormal': 0,
            'input_nonfinite': 0, 'overflow': 0, 'output_nonfinite': 0,
            'max_absolute_error': 0.0, 'max_relative_error': 0.0, 'sum_absolute_error': 0.0}

def reference_verify_blocks(source_raw, actual_raw, statistics):
    demand(len(source_raw) == len(actual_raw) and len(source_raw) % 2 == 0, 'block length mismatch')
    for (source_bits,), (actual_bits,) in zip(struct.iter_unpack('<H', source_raw), struct.iter_unpack('<H', actual_raw)):
        expected = expected_half_bits(source_bits)
        demand(actual_bits == expected, 'numerical FP16 bit mismatch')
        source_value = reference_value(source_bits)
        # Decode actual F16 using independent exponent arithmetic.
        exponent = (actual_bits >> 10) & 31
        fraction = actual_bits & 1023
        magnitude = math.ldexp(fraction, -24) if exponent == 0 else math.ldexp(1024+fraction, exponent-25)
        actual_value = -magnitude if actual_bits & 0x8000 else magnitude
        error = abs(actual_value-source_value)
        statistics['elements'] += 1
        statistics['changed'] += actual_value != source_value
        statistics['zeroed_nonzero'] += source_value != 0 and actual_value == 0
        statistics['output_subnormal'] += 0 < (actual_bits & 0x7fff) < 0x400
        statistics['max_absolute_error'] = max(statistics['max_absolute_error'], error)
        statistics['max_relative_error'] = max(statistics['max_relative_error'], error/abs(source_value) if source_value else 0.0)
        statistics['sum_absolute_error'] += error

def verify_blocks(source_raw, actual_raw, statistics):
    from native_engine import ACTIVE
    worker = ACTIVE.get()
    if worker is None:
        return reference_verify_blocks(source_raw, actual_raw, statistics)
    worker.block(source_raw, statistics, actual_raw)


def verify_directories(source, output, anchor):
    before = source.snapshot()
    out_before = output.snapshot()
    _, source_manifest, source_index, source_headers, count, config = validate_source(source, anchor)
    out_manifest, out_manifest_raw = checked_manifest(output, OUTPUT_MANIFEST)
    demand(out_manifest.get('schema') == 'axonai.fp16-output-sha256.v1'
           and out_manifest.get('self_hash_excluded') is True, 'invalid output manifest schema')
    expected_files = set(COPY_FILES) | set(source_headers) | {'config.json', 'model.safetensors.index.json',
                     'SOURCE_SHA256_MANIFEST.json', 'CONVERSION.json', OUTPUT_MANIFEST}
    demand(output.names() == expected_files, 'derived artifact inventory mismatch')
    demand(output.read('SOURCE_SHA256_MANIFEST.json') == source_manifest, 'source manifest receipt mismatch')
    for name in COPY_FILES + ('model.safetensors.index.json',):
        # Byte equality through SHA256, including the full tokenizer and template.
        demand(source.fingerprint(name) == output.fingerprint(name), 'passthrough metadata mismatch: '+name)
    expected_config = dict(config)
    expected_config['dtype'] = 'float16'
    if 'torch_dtype' in expected_config:
        expected_config['torch_dtype'] = 'float16'
    demand(output.read('config.json') == canonical(expected_config), 'config consistency mismatch')
    out_index, out_headers, out_count = model_metadata(output, 'F16')
    demand(out_index == source_index and out_count == count, 'output model metadata mismatch')
    statistics = {}
    for name, (header, offset, _) in sorted(source_headers.items()):
        actual_header, actual_offset, _ = out_headers[name]
        expected_header = {k: (dict(v, dtype='F16') if k != '__metadata__' else v) for k,v in header.items()}
        demand(actual_header == expected_header, 'tensor metadata mismatch')
        with source.reader(name) as src, output.reader(name) as dst:
            src.seek(offset)
            dst.seek(actual_offset)
            for key, entry in sorted(((k,v) for k,v in header.items() if k != '__metadata__'),
                                      key=lambda kv: kv[1]['data_offsets'][0]):
                remaining = entry['data_offsets'][1]-entry['data_offsets'][0]
                st = initial_statistics()
                while remaining:
                    length = min(CHUNK, remaining)
                    raw = src.read(length)
                    actual = dst.read(length)
                    demand(len(raw) == length and len(actual) == length, 'tensor truncated during verification')
                    verify_blocks(raw, actual, st)
                    remaining -= length
                statistics[key] = st
            demand(not src.read(1) and not dst.read(1), 'trailing payload during verification')
    conversion = parse_json(output.read('CONVERSION.json'))
    demand(conversion.get('schema') == 'axonai.fp16-conversion.v1'
           and conversion.get('source_manifest_sha256') == anchor
           and conversion.get('statistics') == statistics
           and conversion.get('parameters') == count
           and conversion.get('tensor_count') == len(source_index['weight_map'])
           and conversion.get('source_dtype') == 'BF16' and conversion.get('output_dtype') == 'F16'
           and conversion.get('semantics') == 'IEEE754-binary16-roundTiesToEven-preserve-signed-zero-no-clamping'
           and conversion.get('inference_equivalence_established') is False, 'conversion receipt mismatch')
    __import__('native_engine').validate_provenance(conversion.get('candidate_engine'))
    # Converter provenance must match this reviewed code package.
    with Directory(os.path.dirname(os.path.abspath(__file__))) as tools:
        expected_tools = {n: tools.fingerprint(n)['sha256'] for n in ('audit_io.py', 'convert_fp16.py', 'verify_fp16.py')}
    demand(conversion.get('tool_sha256') == expected_tools, 'converter/verifier code identity mismatch')
    # Rehash BOTH sides after numerical inspection; detect concurrent changes.
    validate_source(source, anchor)
    checked_manifest(output, OUTPUT_MANIFEST, sha_bytes(out_manifest_raw))
    demand(source.snapshot() == before and output.snapshot() == out_before, 'files changed during verification')
    return {'status': 'PASS', 'schema': 'axonai.fp16-verification.v1', 'source_manifest_sha256': anchor,
            'output_manifest_sha256': sha_bytes(out_manifest_raw), 'tensor_count': len(source_index['weight_map']),
            'parameters': count, 'source_integrity_rechecked': True, 'complete_output_inventory_verified': True,
            'numerical_result': 'every output bit equals independently computed BF16-to-FP16 roundTiesToEven',
            'inference_equivalence_established': False, 'tool_sha256': expected_tools,
            'zeroed_nonzero': sum(v['zeroed_nonzero'] for v in statistics.values()),
            'output_subnormal': sum(v['output_subnormal'] for v in statistics.values())}

def verify(source_path, output_path, anchor=SOURCE_SHA):
    from pathlib import Path
    workspace = Path(__file__).absolute().parent / 'test-workspace'
    for path in (source_path, output_path):
        demand(Path(path).absolute().is_relative_to(workspace), 'candidate synthetic workspace only')
    with Directory(source_path) as source, Directory(output_path) as output:
        return verify_directories(source, output, anchor)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-name', required=True)
    p.add_argument('--receipt', help='new external receipt file; parent must already exist')
    a = p.parse_args()
    try:
        from audit_io import basename
        basename(a.artifact_name)
        demand(a.receipt is not None, 'external receipt path required before processing')
        with ReceiptDestination(a.receipt) as destination:
            result = verify(SOURCE, os.path.join(CONVERTED, a.artifact_name))
            try:
                destination.publish(result)
            except BaseException as exc:
                raise LifecycleError(dict(result,phase='receipt_write',numerical_verification='PASS',receipt_durability='uncertain'),exc) from exc
        return 0
    except BaseException as exc:
        result = {'status': 'FAIL', 'phase': 'verification', 'error': str(exc), 'inference_equivalence_established': False}
        # FAIL is always machine-readable on stdout, even if receipt path is unsafe.
        report=exc.state if isinstance(exc,LifecycleError) else result
        __import__('native_engine').record_worker_state(report, exc)
        emit_receipt(report)
        return 1

if __name__ == '__main__':
    sys.exit(main())
