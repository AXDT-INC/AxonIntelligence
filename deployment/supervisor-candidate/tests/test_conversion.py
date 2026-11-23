"""Synthetic fixtures only. No production model access, ML packages or GPUs."""
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
from audit_io import (AuditError, COPY_FILES, Directory, MANIFEST, OUTPUT_MANIFEST,
                      canonical, checked_manifest, make_output_manifest, sha_bytes)
from convert_fp16 import convert, convert_block, empty_stats
from verify_fp16 import expected_half_bits, verify

# Defense in depth: a mistaken production default or GPU access must fail before I/O.
def deny_production(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0])
        if path.startswith(('/data/axonai/models/', '/dev/nvidia', '/dev/dri')):
            raise AssertionError('tests must never open production model or GPU paths')
sys.addaudithook(deny_production)

# Directory opens use dir_fd and therefore audit events contain relative component
# names. Guard its entry point as well, before it can open a production directory.
_original_directory_init = Directory.__init__
def fixture_directory_init(self, path=None, *, fd=None):
    candidate = os.path.abspath(os.fspath(path)) if path is not None else os.readlink('/proc/self/fd/'+str(fd))
    if candidate.startswith(('/data/axonai/models', '/dev/nvidia', '/dev/dri')):
        raise AssertionError('production/GPU Directory forbidden by synthetic test guard')
    return _original_directory_init(self, path, fd=fd)
Directory.__init__ = fixture_directory_init

def write_shard(path, items):
    header = {'__metadata__': {'format': 'pt'}}
    data = bytearray()
    for name, bits in items:
        start = len(data)
        data.extend(struct.pack('<'+'H'*len(bits), *bits))
        header[name] = {'dtype': 'BF16', 'shape': [len(bits)], 'data_offsets': [start, len(data)]}
    raw = canonical(header).rstrip(b'\n')
    raw += b' ' * (-len(raw) % 8)
    path.write_bytes(struct.pack('<Q', len(raw))+raw+data)

def rewrite_header(path, change):
    data = path.read_bytes()
    length = struct.unpack('<Q', data[:8])[0]
    header = json.loads(data[8:8+length])
    change(header)
    raw = canonical(header).rstrip(b'\n')
    raw += b' ' * (-len(raw) % 8)
    path.write_bytes(struct.pack('<Q', len(raw))+raw+data[8+length:])

def source_manifest(source):
    files = []
    for p in sorted(source.iterdir()):
        if p.name != MANIFEST:
            b = p.read_bytes()
            files.append({'relative_path': p.name, 'bytes': len(b), 'sha256': sha_bytes(b)})
    m = {'schema': 'axonai.standalone-model-sha256.v1', 'files': files,
         'payload_bytes': sum(x['bytes'] for x in files), 'self_hash_excluded': True,
         'weight_shards': len(list(source.glob('*.safetensors')))}
    data = canonical(m)
    (source/MANIFEST).write_bytes(data)
    return sha_bytes(data)

def update_output_manifest(output):
    (output/OUTPUT_MANIFEST).unlink()
    with Directory(str(output)) as d:
        d.write(OUTPUT_MANIFEST, canonical(make_output_manifest(d)))

class ConversionTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.temporary = tempfile.TemporaryDirectory(prefix='fixture-', dir=str(ROOT/'test-workspace'))
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root/'source'
        self.source.mkdir()
        self.parent = self.root/'converted'
        self.parent.mkdir()
        for name in COPY_FILES:
            (self.source/name).write_bytes(b'{}\n' if name.endswith('.json') else b'synthetic fixture only\n')
        config = {'architectures': ['Qwen3_5ForCausalLM'], 'model_type': 'qwen3_5_text',
                  'dtype': 'bfloat16', 'mamba_ssm_dtype': 'float32', 'untouched_setting': [1,2,3]}
        (self.source/'config.json').write_bytes(canonical(config))
        self.shard = self.source/'model-00001-of-00001.safetensors'
        self.set_tensors([('a', [0, 0x8000, 0x3f80, 0xbf80, 0x477f, 0x3880, 0x3380, 0x3300, 0x3301, 1, 0x8001]),
                          ('b', [0x3f00, 0x3eaa])])

    def set_tensors(self, tensors):
        write_shard(self.shard, tensors)
        count = sum(len(values) for _, values in tensors)
        index = {'metadata': {'total_size': count*2, 'total_parameters': count},
                 'weight_map': {k:self.shard.name for k,_ in tensors}}
        (self.source/'model.safetensors.index.json').write_bytes(canonical(index))
        self.anchor = source_manifest(self.source)

    def run_conversion(self, name='fp16'):
        return convert(str(self.source), str(self.parent), name, self.anchor, reserve_bytes=0)

    def assert_source_failure(self):
        with self.assertRaises((AuditError, OSError, ValueError, TypeError)):
            self.run_conversion()
        self.assertFalse((self.parent/'fp16').exists())

    def test_production_directory_blocked_before_descriptor_open(self):
        with patch('audit_io.os.open', side_effect=AssertionError('os.open must not run')) as opened:
            with self.assertRaisesRegex(AssertionError, 'synthetic test guard'):
                Directory('/data/axonai/models/merged/AxonAI-v1-27B')
            opened.assert_not_called()

    def test_correct_conversion_and_source_immutability(self):
        with Directory(str(self.source)) as d:
            before = d.snapshot()
            hashes = {n:d.fingerprint(n) for n in d.names()}
        result = self.run_conversion()
        self.assertEqual(result['status'], 'PASS')
        self.assertTrue(result['promoted'])
        output = self.parent/'fp16'
        self.assertEqual(verify(str(self.source), str(output), self.anchor)['status'], 'PASS')
        config = json.loads((output/'config.json').read_bytes())
        self.assertEqual(config['dtype'], 'float16')
        self.assertEqual(config['mamba_ssm_dtype'], 'float32')
        self.assertEqual(config['untouched_setting'], [1,2,3])
        with Directory(str(self.source)) as d:
            self.assertEqual(before, d.snapshot())
            self.assertEqual(hashes, {n:d.fingerprint(n) for n in d.names()})

    def test_all_65536_bf16_patterns_against_independent_reference(self):
        for bits in range(65536):
            magnitude = bits & 0x7fff
            raw = struct.pack('<H', bits)
            if magnitude > 0x477f:
                with self.assertRaises(AuditError):
                    convert_block(raw, empty_stats())
                with self.assertRaises(AuditError):
                    expected_half_bits(bits)
            else:
                actual = struct.unpack('<H', convert_block(raw, empty_stats()))[0]
                self.assertEqual(actual, expected_half_bits(bits), hex(bits))
        self.assertEqual(expected_half_bits(0x8000), 0x8000)
        self.assertEqual(expected_half_bits(0x3380), 1)  # 2^-24
        self.assertEqual(expected_half_bits(0x3300), 0)  # tie to even zero
        self.assertEqual(expected_half_bits(0x3301), 1)  # above midpoint

    def test_underflow_is_recorded_not_clamped(self):
        self.set_tensors([('a', [1, 0x8001, 0x3300, 0x3380])])
        self.run_conversion()
        st = json.loads((self.parent/'fp16'/'CONVERSION.json').read_bytes())['statistics']['a']
        self.assertEqual(st['zeroed_nonzero'], 3)
        self.assertEqual(st['output_subnormal'], 1)
        self.assertEqual(st['changed'], 3)
        self.assertGreater(st['max_absolute_error'], 0)

    def test_overflow_rejected_and_partial_staging_retained(self):
        self.set_tensors([('a', [0x3f80]), ('z', [0x4780])])
        self.assert_source_failure()
        stages = list(self.parent.glob('.fp16.staging-*'))
        self.assertEqual(len(stages), 1)
        self.assertFalse((stages[0]/OUTPUT_MANIFEST).exists())
        self.assertTrue((stages[0]/self.shard.name).exists())

    def test_nonfinite_inputs_rejected(self):
        for bits in (0x7f80, 0xff80, 0x7fc0, 0x7f81, 0xffc0):
            with self.subTest(bits=bits):
                self.set_tensors([('a', [bits])])
                self.assert_source_failure()

    def test_incorrect_source_anchor(self):
        self.anchor = '0'*64
        self.assert_source_failure()
        self.assertEqual(list(self.parent.iterdir()), [])

    def test_hash_and_size_mismatch(self):
        with self.shard.open('ab') as f:
            f.write(b'corruption')
        self.assert_source_failure()

    def test_unexpected_source_file(self):
        (self.source/'unlisted.txt').write_text('not authorized')
        self.assert_source_failure()

    def test_missing_source_file(self):
        (self.source/'chat_template.jinja').unlink()
        self.assert_source_failure()

    def test_truncated_and_trailing_safetensors_even_with_updated_manifest(self):
        original = self.shard.read_bytes()
        for data in (b'', b'1234567', struct.pack('<Q', 100000000), original[:-1], original+b'x'):
            with self.subTest(size=len(data)):
                self.shard.write_bytes(data)
                self.anchor = source_manifest(self.source)
                self.assert_source_failure()

    def test_wrong_dtype_shape_offsets(self):
        original = self.shard.read_bytes()
        for field, value in [('dtype','F32'), ('shape',[999]), ('data_offsets',[2,22])]:
            self.shard.write_bytes(original)
            rewrite_header(self.shard, lambda h:h['a'].__setitem__(field,value))
            self.anchor = source_manifest(self.source)
            self.assert_source_failure()

    def test_tensor_index_mismatch(self):
        p = self.source/'model.safetensors.index.json'
        index = json.loads(p.read_bytes())
        index['weight_map']['missing'] = self.shard.name
        p.write_bytes(canonical(index))
        self.anchor = source_manifest(self.source)
        self.assert_source_failure()

    def test_manifest_duplicate_and_path_traversal(self):
        p = self.source/MANIFEST
        original = json.loads(p.read_bytes())
        for unsafe in ('../escape', '/absolute', 'sub/file', 'sub\\file', '..', '.'):
            m = json.loads(canonical(original))
            m['files'][0]['relative_path'] = unsafe
            data = canonical(m)
            p.write_bytes(data)
            self.anchor = sha_bytes(data)
            self.assert_source_failure()
        m = original
        m['files'].append(m['files'][0])
        data = canonical(m)
        p.write_bytes(data)
        self.anchor = sha_bytes(data)
        self.assert_source_failure()

    def test_duplicate_json_key_rejected(self):
        data = b'{"files":[],"files":[]}'
        (self.source/MANIFEST).write_bytes(data)
        self.anchor = sha_bytes(data)
        self.assert_source_failure()

    def test_symlink_file_and_source_ancestor(self):
        real = self.root/'real-shard'
        self.shard.rename(real)
        self.shard.symlink_to(real)
        self.assert_source_failure()
        self.shard.unlink()
        real.rename(self.shard)
        link = self.root/'source-link'
        link.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(OSError):
            convert(str(link), str(self.parent), 'fp16', self.anchor, 0)

    def test_hardlinked_source_rejected(self):
        os.link(self.shard, self.root/'hardlink')
        self.assert_source_failure()

    def test_output_parent_symlink_rejected(self):
        link = self.root/'parent-link'
        link.symlink_to(self.parent, target_is_directory=True)
        with self.assertRaises(OSError):
            convert(str(self.source), str(link), 'fp16', self.anchor, 0)

    def test_existing_output_and_symlink_never_overwritten(self):
        out = self.parent/'fp16'
        out.mkdir()
        (out/'sentinel').write_text('keep')
        self.assert_source_failure_existing(out)
        (out/'sentinel').unlink()
        out.rmdir()
        out.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(AuditError):
            self.run_conversion()
        self.assertTrue(out.is_symlink())

    def assert_source_failure_existing(self, out):
        with self.assertRaises(AuditError):
            self.run_conversion()
        self.assertEqual((out/'sentinel').read_text(), 'keep')

    def test_output_name_traversal(self):
        for name in ('../escape', '/escape', 'sub/file', '.', '..'):
            with self.assertRaises(AuditError):
                self.run_conversion(name)

    def test_insufficient_storage(self):
        class Space:
            f_bavail = 0
            f_frsize = 4096
            f_favail = 100
            f_files = 100
        with patch('convert_fp16.os.fstatvfs', return_value=Space()):
            self.assert_source_failure()
        self.assertEqual(list(self.parent.iterdir()), [])

    def test_partial_io_failure_never_published(self):
        with patch('convert_fp16.convert_block', side_effect=OSError('synthetic disk failure')):
            self.assert_source_failure()
        self.assertEqual(len(list(self.parent.glob('.fp16.staging-*'))), 1)

    def test_promotion_race_never_overwrites(self):
        import convert_fp16
        original = convert_fp16.promote
        def collision(parent, stage, final):
            os.mkdir(final, dir_fd=parent.fd)
            return original(parent, stage, final)
        with patch('convert_fp16.promote', side_effect=collision):
            with self.assertRaises(AuditError):
                self.run_conversion()
        self.assertEqual(list((self.parent/'fp16').iterdir()), [])
        self.assertEqual(len(list(self.parent.glob('.fp16.staging-*'))), 1)

    def test_numerical_corruption_rejected_even_after_rehash(self):
        self.run_conversion()
        output = self.parent/'fp16'
        shard = output/self.shard.name
        data = bytearray(shard.read_bytes())
        data[-1] ^= 1
        shard.write_bytes(data)
        update_output_manifest(output)
        with self.assertRaisesRegex(AuditError, 'numerical FP16 bit mismatch'):
            verify(str(self.source), str(output), self.anchor)

    def test_output_metadata_and_inventory(self):
        self.run_conversion()
        output = self.parent/'fp16'
        p = output/'config.json'
        cfg = json.loads(p.read_bytes())
        cfg['mamba_ssm_dtype'] = 'float16'
        p.write_bytes(canonical(cfg))
        update_output_manifest(output)
        with self.assertRaisesRegex(AuditError, 'config consistency'):
            verify(str(self.source), str(output), self.anchor)
        (output/'extra').write_text('no')
        with self.assertRaises(AuditError):
            verify(str(self.source), str(output), self.anchor)

    def test_output_dtype_rejected_after_rehash(self):
        self.run_conversion()
        output = self.parent/'fp16'
        rewrite_header(output/self.shard.name, lambda h:h['a'].__setitem__('dtype','BF16'))
        update_output_manifest(output)
        with self.assertRaises(AuditError):
            verify(str(self.source), str(output), self.anchor)

    def test_receipt_tampering_rejected_after_rehash(self):
        self.run_conversion()
        output = self.parent/'fp16'
        p = output/'CONVERSION.json'
        data = json.loads(p.read_bytes())
        data['statistics']['a']['zeroed_nonzero'] = 999
        p.write_bytes(canonical(data))
        update_output_manifest(output)
        with self.assertRaisesRegex(AuditError, 'conversion receipt mismatch'):
            verify(str(self.source), str(output), self.anchor)

    def test_deterministic_artifacts_and_verification_receipts(self):
        self.run_conversion('one')
        self.run_conversion('two')
        first = verify(str(self.source), str(self.parent/'one'), self.anchor)
        second = verify(str(self.source), str(self.parent/'two'), self.anchor)
        self.assertEqual(canonical(first), canonical(second))
        for p in (self.parent/'one').iterdir():
            self.assertEqual(p.read_bytes(), (self.parent/'two'/p.name).read_bytes())
        self.assertFalse(first['inference_equivalence_established'])

    def test_source_mutation_after_conversion_detected(self):
        self.run_conversion()
        (self.source/'chat_template.jinja').write_text('changed synthetic source')
        with self.assertRaises(AuditError):
            verify(str(self.source), str(self.parent/'fp16'), self.anchor)

    def test_empty_tensors_and_chunk_boundaries(self):
        from audit_io import CHUNK
        # Empty tensor sorts after nonempty by name, but before it by offsets.
        self.set_tensors([('z_empty', []), ('a_data', [0x3f80]*(CHUNK//2+3))])
        self.run_conversion()
        result = verify(str(self.source), str(self.parent/'fp16'), self.anchor)
        self.assertEqual(result['parameters'], CHUNK//2+3)

    def test_multiple_shards_and_duplicate_tensor_detection(self):
        first = self.source/'model-00001-of-00002.safetensors'
        self.shard.rename(first)
        self.shard = first
        second = self.source/'model-00002-of-00002.safetensors'
        write_shard(second, [('extra',[0x3f80])])
        p = self.source/'model.safetensors.index.json'
        index = json.loads(p.read_bytes())
        index['weight_map'] = {k:first.name for k in index['weight_map']}
        index['weight_map']['extra'] = second.name
        index['metadata']['total_parameters'] += 1
        index['metadata']['total_size'] += 2
        p.write_bytes(canonical(index))
        self.anchor = source_manifest(self.source)
        self.run_conversion()
        self.assertEqual(verify(str(self.source), str(self.parent/'fp16'), self.anchor)['tensor_count'], 3)
        write_shard(second, [('a',[0x3f80])])
        self.anchor = source_manifest(self.source)
        with self.assertRaises(AuditError):
            self.run_conversion('duplicate')
        self.assertFalse((self.parent/'duplicate').exists())

    def test_output_truncation_rejected_even_after_rehash(self):
        self.run_conversion()
        output = self.parent/'fp16'
        p = output/self.shard.name
        p.write_bytes(p.read_bytes()[:-1])
        update_output_manifest(output)
        with self.assertRaisesRegex(AuditError, 'truncated'):
            verify(str(self.source), str(output), self.anchor)

    def test_tokenizer_and_template_changes_rejected_after_rehash(self):
        self.run_conversion()
        output = self.parent/'fp16'
        for name in ('tokenizer.json', 'chat_template.jinja'):
            p = output/name
            original = p.read_bytes()
            p.write_bytes(original+b'changed')
            update_output_manifest(output)
            with self.assertRaisesRegex(AuditError, 'passthrough metadata mismatch'):
                verify(str(self.source), str(output), self.anchor)
            p.write_bytes(original)
            update_output_manifest(output)

    def test_cpu_cli_requires_explicit_authorization_before_source_access(self):
        from convert_fp16 import main
        from contextlib import redirect_stdout
        from io import StringIO
        text = StringIO()
        with patch('sys.argv', ['convert_fp16.py', '--artifact-name','unused']), redirect_stdout(text):
            result = main()
        self.assertEqual(result, 1)
        self.assertEqual(json.loads(text.getvalue())['status'], 'FAIL')

if __name__ == '__main__':
    unittest.main()
