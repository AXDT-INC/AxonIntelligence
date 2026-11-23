#!/usr/bin/env python3
"""Offline CPU streaming converter. CLI is production anchored; tests call convert()."""
import argparse
import ctypes
import errno
import math
import os
import struct
import sys
import uuid
from audit_io import (AuditError, CHUNK, CONVERTED, COPY_FILES, Directory, OUTPUT_MANIFEST,
                      SOURCE, SOURCE_SHA, basename, canonical, copy_file, demand, emit_receipt,
                      make_output_manifest, output_config, sha_bytes, validate_source, ReceiptDestination, LifecycleError, write_all)

TOOLS = ('audit_io.py', 'convert_fp16.py', 'verify_fp16.py')

def tool_hashes():
    with Directory(os.path.dirname(os.path.abspath(__file__))) as d:
        return {n: d.fingerprint(n)['sha256'] for n in TOOLS}

def empty_stats():
    return dict(elements=0, changed=0, zeroed_nonzero=0, output_subnormal=0,
                input_nonfinite=0, overflow=0, output_nonfinite=0,
                max_absolute_error=0.0, max_relative_error=0.0, sum_absolute_error=0.0)

def reference_convert_block(raw, stats):
    """BF16 bits -> exact binary32 -> binary16, CPython struct IEEE ties-to-even.

    Reject every nonfinite source and magnitude > 65504. Preserve signed zero.
    Subnormals are retained; rounding to signed zero is counted, never replaced.
    """
    demand(len(raw) % 2 == 0, 'odd tensor block')
    result = bytearray(len(raw))
    for i, (bits,) in enumerate(struct.iter_unpack('<H', raw)):
        value = struct.unpack('<f', struct.pack('<I', bits << 16))[0]
        demand(math.isfinite(value), 'nonfinite BF16 input')
        demand(abs(value) <= 65504.0, 'FP16 overflow')
        try:
            packed = struct.pack('<e', value)
        except OverflowError as exc:
            raise AuditError('FP16 overflow') from exc
        actual = struct.unpack('<e', packed)[0]
        demand(math.isfinite(actual), 'nonfinite FP16 output')
        result[2*i:2*i+2] = packed
        out_bits = int.from_bytes(packed, 'little')
        error = abs(actual-value)
        stats['elements'] += 1
        stats['changed'] += actual != value
        stats['zeroed_nonzero'] += value != 0.0 and actual == 0.0
        stats['output_subnormal'] += 0 < (out_bits & 0x7fff) < 0x400
        stats['max_absolute_error'] = max(stats['max_absolute_error'], error)
        stats['max_relative_error'] = max(stats['max_relative_error'], error/abs(value) if value else 0.0)
        stats['sum_absolute_error'] += error
    return result

def convert_block(raw, stats):
    from native_engine import ACTIVE
    worker = ACTIVE.get()
    return reference_convert_block(raw, stats) if worker is None else worker.block(raw, stats)


def check_space(parent, needed):
    s = os.fstatvfs(parent.fd)
    demand(s.f_bavail*s.f_frsize >= needed, 'insufficient storage')
    demand(s.f_files == 0 or s.f_favail >= 64, 'insufficient free inodes')

def load_publication_gate(worker):
    """Authenticate the image bytes before loading them into this coordinator."""
    import fcntl
    import hashlib
    import native_engine
    with Directory(str(native_engine.ROOT)) as directory:
        image = directory.read('publication_gate.so', 1024 * 1024)
    demand(hashlib.sha256(image).hexdigest() == worker.trust['sources']['publication_gate.so'],
           'publication library hash mismatch')
    fd = os.memfd_create('axon-publication', os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
    with os.fdopen(fd, 'wb', buffering=0) as stream:
        write_all(stream, image)
        seals = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS, seals)
        demand(fcntl.fcntl(fd, fcntl.F_GET_SEALS) == seals, 'publication library seal mismatch')
        library = ctypes.PyDLL('/proc/self/fd/' + str(fd), use_errno=True)
        pointer=ctypes.c_void_p; word=ctypes.c_uint64
        library.gate_create.argtypes=[word,word];library.gate_create.restype=pointer
        for name in ('gate_arm','gate_invalidate'):
            fn=getattr(library,name);fn.argtypes=[pointer,word,word];fn.restype=ctypes.c_int
        library.gate_status.argtypes=[pointer,word,word,ctypes.POINTER(word)];library.gate_status.restype=ctypes.c_int
        library.publish_once.argtypes=[ctypes.c_int,pointer,word,word,ctypes.c_char_p,ctypes.c_char_p]
        library.publish_once.restype=ctypes.c_int
        return library

def promote(parent, stage, final):
    """Atomic Linux rename with RENAME_NOREPLACE; never use overwrite-capable rename."""
    import native_engine
    worker = native_engine.ACTIVE.get()
    if worker is not None:
        demand(worker.commit_granted and worker.coordinator_phase == 'rename_attempted'
               and worker.rename_target == (parent, stage, final), 'unbound promotion')
        library = worker.publication_library
        fn = library.publish_once
        if fn(parent.fd, worker.publication_gate, worker.gate_session, worker.gate_generation,
              stage.encode(), final.encode()):
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        return
    libc = ctypes.CDLL(None, use_errno=True)
    fn = getattr(libc, 'renameat2', None)
    demand(fn is not None, 'atomic no-replace rename unavailable; staging retained')
    fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    fn.restype = ctypes.c_int
    if fn(parent.fd, stage.encode(), parent.fd, final.encode(), 1):
        code = ctypes.get_errno()
        error = OSError(code, os.strerror(code))
        error.reference_rename_rejected = True
        raise error

def record_promotion_oserror(worker, state, error):
    """Resolve this attempt from its gate, leaving ambiguous exceptions UNKNOWN."""
    if worker is None:
        if not getattr(error, 'reference_rename_rejected', False):
            return
        promoted = False
    else:
        try:
            # Close the attempt before interpreting an unattempted record.
            worker.invalidate_native_publication()
            promoted = bool(worker.native_publication_status()['confirmed'])
        except Exception as observation_error:
            worker.diagnostic('observe_promotion_error', observation_error)
            return
    state.update(promoted=promoted, durability='uncertain' if promoted else 'not_published')

def convert(source_path, parent_path, final_name, anchor=SOURCE_SHA, reserve_bytes=4*1024**3, lifecycle=None):
    # Candidate is deliberately restricted to synthetic fixtures in this workspace.
    from pathlib import Path
    workspace = Path(__file__).absolute().parent / 'test-workspace'
    for path in (source_path, parent_path):
        demand(Path(path).absolute().is_relative_to(workspace), 'candidate synthetic workspace only')
    state = lifecycle if lifecycle is not None else {}
    state.update(phase='preflight',promoted=False,durability='not_published')
    basename(final_name)
    demand(not final_name.startswith('.'), 'final name must not be hidden')
    with Directory(source_path) as source, Directory(parent_path) as parent:
        try:
            os.stat(final_name, dir_fd=parent.fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise AuditError('output already exists')
        # No staging or output files are created before all source checks succeed.
        before = source.snapshot()
        manifest, manifest_raw, index, headers, count, config = validate_source(source, anchor)
        demand(source.snapshot() == before, 'source changed during initial validation')
        demand(manifest['payload_bytes'] <= 16*1024**2 + 1024**2, 'candidate fixture size limit')
        demand(type(reserve_bytes) is int and reserve_bytes >= 0, 'invalid reserve')
        needed = manifest['payload_bytes'] + 16*1024**2 + reserve_bytes
        check_space(parent, needed)
        stage_name = '.' + final_name + '.staging-' + uuid.uuid4().hex
        # Failure deliberately retains a private staging directory, without promotion.
        state.update(phase='conversion', artifact_name=final_name, staging_name=stage_name, promoted=False, durability='not_published', numerical_verification='not_completed')
        try:
            with parent.mkdir(stage_name) as stage:
                statistics = {}
                for shard, (header, offset, _) in sorted(headers.items()):
                    new_header = {k: (dict(v, dtype='F16') if k != '__metadata__' else v)
                                  for k,v in header.items()}
                    packed = canonical(new_header).rstrip(b'\n')
                    packed += b' ' * (-len(packed) % 8)
                    with source.reader(shard) as src, stage.writer(shard) as dst:
                        write_all(dst,struct.pack('<Q', len(packed)))
                        write_all(dst,packed)
                        src.seek(offset)
                        for name, entry in sorted(((k,v) for k,v in header.items() if k != '__metadata__'),
                                                  key=lambda kv: kv[1]['data_offsets'][0]):
                            remaining = entry['data_offsets'][1]-entry['data_offsets'][0]
                            stats = empty_stats()
                            while remaining:
                                raw = src.read(min(CHUNK, remaining))
                                demand(len(raw) == min(CHUNK, remaining), 'source truncated during conversion')
                                write_all(dst,convert_block(raw, stats))
                                remaining -= len(raw)
                            statistics[name] = stats
                        demand(not src.read(1), 'source grew during conversion')
                        dst.flush()
                        os.fsync(dst.fileno())
                for name in COPY_FILES:
                    copy_file(source, stage, name)
                # Index sizes do not change: both BF16 and F16 use two bytes.
                copy_file(source, stage, 'model.safetensors.index.json')
                stage.write('config.json', canonical(output_config(config)))
                stage.write('SOURCE_SHA256_MANIFEST.json', manifest_raw)
                conversion = {'schema': 'axonai.fp16-conversion.v1', 'source_manifest_sha256': anchor,
                              'semantics': 'IEEE754-binary16-roundTiesToEven-preserve-signed-zero-no-clamping',
                              'source_dtype': 'BF16', 'output_dtype': 'F16',
                              'tensor_count': len(index['weight_map']), 'parameters': count,
                              'statistics': statistics, 'tool_sha256': tool_hashes(),
                              'candidate_engine': __import__('native_engine').provenance(),
                              'inference_equivalence_established': False}
                stage.write('CONVERSION.json', canonical(conversion))
                stage.write(OUTPUT_MANIFEST, canonical(make_output_manifest(stage)))
                # Separate verifier reopens source/output files and uses integer bit arithmetic.
                from verify_fp16 import verify_directories
                state['phase']='numerical_verification'
                result = verify_directories(source, stage, anchor)
                state['numerical_verification']='PASS'
                __import__('native_engine').finish_before_publication(result, state)
                state.update(phase='pre_publication',numerical_verification='PASS')
                demand(source.snapshot() == before, 'source identity/content metadata changed')
                os.fsync(stage.fd)
            # Mark the syscall window unknown: an asynchronous interrupt can occur
            # after rename succeeds but before Python records its return value.
            worker = __import__('native_engine').ACTIVE.get()
            if worker is not None:
                demand(worker.rename_target is None, 'promotion target already bound')
                worker.rename_target = (parent, stage_name, final_name)
            __import__('native_engine').require_publication_authorized(state)
            state.update(phase='atomic_promotion',promoted=None,durability='unknown')
            try:
                __import__('native_engine').perform_promotion(state,promote,parent,stage_name,final_name)
            except OSError as error:
                record_promotion_oserror(worker, state, error)
                raise
            state.update(phase='post_promotion_fsync',promoted=True,durability='uncertain')
            __import__('native_engine').publication_observed(state)
            os.fsync(parent.fd)
            state.update(phase='published',durability='confirmed')
            __import__('native_engine').settle_publication(state)
            __import__('native_engine').complete_publication(state)
            return dict(result, **state)
        except BaseException as exc:
            __import__('native_engine').settle_publication(state,failed=True)
            __import__('native_engine').complete_publication(state,exc)
            __import__('native_engine').record_worker_state(state, exc)
            state['primary_failure'] = dict(error=str(exc),error_type=type(exc).__name__,interrupted=isinstance(exc,(KeyboardInterrupt,SystemExit)))
            raise LifecycleError(state,exc) from exc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-name', required=True)
    p.add_argument('--authorize-cpu-conversion', action='store_true')
    p.add_argument('--receipt', help='new external receipt file; parent must already exist')
    a = p.parse_args()
    state = dict(phase='preflight',promoted=False,durability='not_published',inference_equivalence_established=False)
    destination = None
    failure = None
    try:
        demand(a.authorize_cpu_conversion, 'explicit CPU conversion authorization flag required')
        demand(a.receipt is not None, 'external receipt path required before processing')
        destination = ReceiptDestination(a.receipt)
        destination.__enter__()
        result = convert(SOURCE, CONVERTED, a.artifact_name, lifecycle=state)
        state.update(result)
        state.update(phase='receipt_write',receipt_durability='uncertain')
        destination.publish(result)
        state.update(phase='receipt_cleanup',receipt_durability='confirmed')
    except BaseException as exc:
        failure = exc
        if state.get('phase') == 'published' and not isinstance(exc,LifecycleError):
            state['phase']='conversion_cleanup'
        if isinstance(exc,LifecycleError):
            state.update(exc.state)
    finally:
        if destination is not None:
            try:
                destination.__exit__(type(failure) if failure else None,failure,
                                     failure.__traceback__ if failure else None)
            except BaseException as cleanup:
                if failure is None:
                    failure = cleanup
                    state['phase']='receipt_cleanup'
                else:
                    state['cleanup_error_type']=type(cleanup).__name__
                    state['cleanup_error']=str(cleanup)
    if failure is None:
        return 0
    __import__('native_engine').record_worker_state(state, failure)
    state.update(status='FAIL',error=str(failure),error_type=type(failure).__name__,
                 interrupted=isinstance(failure,(KeyboardInterrupt,SystemExit)))
    if 'primary_failure' in state:
        primary = state['primary_failure']
        if not isinstance(failure,LifecycleError):
            state['cleanup_error_type']=type(failure).__name__
            state['cleanup_error']=str(failure)
        state.update(primary)
    if isinstance(failure,LifecycleError):
        state['error_type']=failure.state.get('error_type',state['error_type'])
        state['interrupted']=failure.state.get('interrupted',False)
    try:
        emit_receipt(state)
    except BaseException as reporting:
        # Reporting failure must not replace the original error/state. Best effort
        # stderr fallback; if both streams fail, exit 1 remains the only signal.
        state['reporting_error_type']=type(reporting).__name__
        try:
            sys.stderr.write(canonical(state).decode())
            sys.stderr.flush()
        except BaseException:
            pass
    return 1


if __name__ == '__main__':
    sys.exit(main())
