import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
# Original package-only helpers are imported read-only for baseline regression.
sys.path.append(str(ROOT.parent))
from audit_io import AuditError, Directory, canonical
from audit_wheelhouse import audit
from future_inference import authorization, NoFusedImports, main as inference_main
from validate_draft import check

class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='package-', dir=str(ROOT/'test-workspace'))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def wheel(self, name, version='1.0', dependencies=()):
        path = self.root/f'{name}-{version}-py3-none-any.whl'
        metadata = f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\nRequires-Python: >=3.10\n'
        metadata += ''.join('Requires-Dist: '+dep+'\n' for dep in dependencies)
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr(f'{name}-{version}.dist-info/METADATA', metadata+'\n')
        return path

    def test_offline_dependency_closure_records_all_metadata_and_hashes(self):
        self.wheel('root', dependencies=('child>=1.0', 'foreign; sys_platform == "win32"'))
        self.wheel('child')
        report, lock = audit(str(self.root), 'root==1.0', ['py3-none-any'])
        self.assertEqual([x['name'] for x in report['dependencies']], ['child','root'])
        self.assertEqual(lock.count('--hash=sha256:'), 2)
        self.assertFalse(report['native_library_abi_verified'])
        self.assertEqual(audit(str(self.root), 'root==1.0', ['py3-none-any']), (report,lock))

    def test_missing_transitive_fails_closed(self):
        self.wheel('root', dependencies=('missing>=1',))
        with self.assertRaisesRegex(AuditError, 'missing transitive'):
            audit(str(self.root), 'root==1.0', ['py3-none-any'])

    def test_dependency_version_conflict_and_tag_rejected(self):
        self.wheel('root', dependencies=('child>=2',))
        self.wheel('child')
        with self.assertRaises(AuditError):
            audit(str(self.root), 'root==1.0', ['py3-none-any'])
        with self.assertRaises(AuditError):
            audit(str(self.root), 'root==1.0', ['cp312-cp312-linux_x86_64'])

    def test_unexpected_wheel_rejected(self):
        self.wheel('root')
        self.wheel('unused')
        with self.assertRaisesRegex(AuditError, 'unreferenced'):
            audit(str(self.root), 'root==1.0', ['py3-none-any'])

    def test_cpu_only_offline_compose_lint(self):
        result = check()
        self.assertEqual(result['status'], 'PASS_OFFLINE_DRAFT_LINT')
        self.assertFalse(result['docker_executed'])
        self.assertEqual(result['default_active_services'], [])

    def test_compose_mutation_rejected(self):
        import yaml
        data = yaml.safe_load((ROOT.parent/'compose.yaml').read_text())
        data['services']['inference']['ports'] = ['8000:8000']
        path = self.root/'bad.yaml'
        path.write_text(yaml.safe_dump(data))
        with self.assertRaises(AssertionError):
            check(path)

    def test_gpu_cli_refuses_before_any_ml_import(self):
        import builtins
        from io import StringIO
        from contextlib import redirect_stdout
        real_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'torch','transformers','triton'}:
                self.fail('ML import occurred before authorization')
            return real_import(name, *args, **kwargs)
        output = StringIO()
        with patch('builtins.__import__', side_effect=guarded), patch('sys.argv', ['future_inference.py']), redirect_stdout(output):
            self.assertEqual(inference_main(), 1)
        self.assertEqual(json.loads(output.getvalue())['status'], 'FAIL')

    def test_missing_and_expired_gpu_authorization(self):
        with self.assertRaises(AuditError):
            authorization(str(self.root/'missing.json'), False)
        data = {'authorized':True, 'purpose':'AxonAI-v1 independent FP16 validation',
                'exclusive_gpu_uuids':['GPU-11111111-1111-1111-1111-111111111111',
                                      'GPU-22222222-2222-2222-2222-222222222222'],
                'expires_utc':'2000-01-01T00:00:00Z'}
        path = self.root/'authorization.json'
        path.write_bytes(canonical(data))
        with self.assertRaisesRegex(AuditError, 'expired'):
            authorization(str(path), True)

    def test_kernel_import_guard_without_loading_any_kernel(self):
        guard = NoFusedImports()
        for name in ('triton', 'triton.compiler', 'fla', 'flash_attn', 'kernels', 'causal_conv1d'):
            with self.assertRaises(ImportError):
                guard.find_spec(name)
        self.assertIsNone(guard.find_spec('math'))

    def test_exclusive_receipt_writer_and_symlink_protection(self):
        with Directory(str(self.root)) as d:
            d.write('receipt.json', b'first')
            with self.assertRaises(FileExistsError):
                d.write('receipt.json', b'second')
        (self.root/'link').symlink_to(self.root/'receipt.json')
        with Directory(str(self.root)) as d:
            with self.assertRaises(OSError):
                d.reader('link')
        self.assertEqual((self.root/'receipt.json').read_bytes(), b'first')

if __name__ == '__main__':
    unittest.main()
