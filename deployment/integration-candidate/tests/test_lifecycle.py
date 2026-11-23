import errno
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).absolute().parents[1]
sys.path.insert(0,str(ROOT))
import audit_io as aio
import convert_fp16 as conv
from test_conversion import ConversionTests, source_manifest, rewrite_header

class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'test-workspace')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.receipts=self.root/'receipts';self.receipts.mkdir(mode=0o700)
        p=patch.object(aio,'RECEIPTS',str(self.receipts));p.start();self.addCleanup(p.stop)

    def test_protected_destinations_and_aliases(self):
        for parent in ('source','converted','staging','deployment'):
            d=self.root/parent;d.mkdir()
            with self.assertRaises(aio.AuditError):aio.ReceiptDestination(str(d/'x.json'))
        for path in (str(self.receipts/'..'/'receipts'/'x'),str(self.receipts)+'//x','relative.json'):
            with self.assertRaises(aio.AuditError):aio.ReceiptDestination(path)
        link=self.root/'alias';link.symlink_to(self.receipts)
        with self.assertRaises(aio.AuditError):aio.ReceiptDestination(str(link/'x'))

    def test_missing_unwritable_and_wrong_permissions(self):
        self.receipts.rmdir()
        with self.assertRaises(OSError):aio.ReceiptDestination(str(self.receipts/'x'))
        self.receipts.mkdir(mode=0o500)
        with self.assertRaises(aio.AuditError):aio.ReceiptDestination(str(self.receipts/'x'))
        self.receipts.chmod(0o700)
        with patch('audit_io.os.access',return_value=False):
            with self.assertRaises(aio.AuditError):aio.ReceiptDestination(str(self.receipts/'x'))

    def test_existing_file_directory_symlink_fifo(self):
        target=self.receipts/'x'
        for kind in ('file','directory','symlink','fifo'):
            if kind=='file':target.write_text('keep')
            elif kind=='directory':target.mkdir()
            elif kind=='symlink':target.symlink_to(self.root/'absent')
            else:os.mkfifo(target)
            with self.assertRaises(aio.AuditError):aio.ReceiptDestination(str(target))
            if kind=='directory':target.rmdir()
            else:target.unlink()

    def test_directory_substitution(self):
        with aio.ReceiptDestination(str(self.receipts/'x')) as dest:
            self.receipts.rename(self.root/'old')
            self.receipts.mkdir(mode=0o700)
            with self.assertRaises(aio.AuditError):dest.publish({'status':'PASS'})
            self.assertFalse((self.receipts/'x').exists())
            self.assertFalse((self.root/'old'/'x').exists())

    def test_collision_after_validation(self):
        with aio.ReceiptDestination(str(self.receipts/'x')) as dest:
            (self.receipts/'x').write_text('keep')
            with self.assertRaises(FileExistsError):dest.publish({})
            self.assertEqual((self.receipts/'x').read_text(),'keep')

    def test_short_zero_enospc_writes(self):
        class Short(io.BytesIO):
            def write(self,data):return super().write(bytes(data[:2]))
        f=Short();aio.write_all(f,b'abcdef');self.assertEqual(f.getvalue(),b'abcdef')
        with patch.object(f,'write',return_value=0):
            with self.assertRaises(aio.AuditError):aio.write_all(f,b'x')
        with patch.object(f,'write',side_effect=OSError(errno.ENOSPC,'full')):
            with self.assertRaises(OSError):aio.write_all(f,b'x')

    def test_receipt_fsync_failure_preserves_exclusive_file(self):
        with aio.ReceiptDestination(str(self.receipts/'x')) as dest:
            with patch('audit_io.os.fsync',side_effect=OSError('fsync')):
                with self.assertRaises(OSError):dest.publish({'status':'PASS'})
        self.assertTrue((self.receipts/'x').exists())

    def test_receipt_preflight_precedes_conversion_and_verification(self):
        import verify_fp16
        for module,flag,function in ((conv,['--authorize-cpu-conversion'],'convert'),(verify_fp16,[],'verify')):
            with patch.object(module,function) as process,patch('sys.argv',['tool','--artifact-name','x','--receipt',str(self.root/'bad'),*flag]),patch('sys.stdout',new_callable=io.StringIO):
                self.assertEqual(module.main(),1);process.assert_not_called()

    def test_post_promotion_receipt_failure_state(self):
        result=dict(status='PASS',promoted=True,durability='confirmed',phase='published',numerical_verification='PASS')
        text=io.StringIO()
        with patch.object(conv,'convert',return_value=result),patch.object(aio.ReceiptDestination,'publish',side_effect=[OSError('disk full'),None]) as publish,patch('sys.argv',['tool','--artifact-name','x','--authorize-cpu-conversion','--receipt',str(self.receipts/'x')]),patch('sys.stdout',text):
            self.assertEqual(conv.main(),1)
            state=publish.call_args.args[0]
            self.assertTrue(state['promoted']);self.assertEqual(state['phase'],'receipt_write')
            self.assertEqual(state['durability'],'confirmed')

class LifecycleTests(unittest.TestCase):
    setUp=ConversionTests.setUp
    set_tensors=ConversionTests.set_tensors
    run_conversion=ConversionTests.run_conversion
    # Inherit fixture helpers, but inherited tests are also intentionally rerun.
    def test_before_promotion_interrupt(self):
        with patch('verify_fp16.verify_directories',side_effect=KeyboardInterrupt()):
            with self.assertRaises(aio.LifecycleError) as caught:self.run_conversion()
        self.assertFalse(caught.exception.state['promoted'])
        self.assertFalse((self.parent/'fp16').exists())

    def test_after_promotion_interrupt(self):
        original=conv.promote
        def interrupt(*args):
            original(*args)
            raise KeyboardInterrupt()
        with patch.object(conv,'promote',side_effect=interrupt):
            with self.assertRaises(aio.LifecycleError) as caught:self.run_conversion()
        self.assertIsNone(caught.exception.state['promoted'])
        self.assertTrue((self.parent/'fp16').exists())
        self.assertEqual(caught.exception.state['numerical_verification'],'PASS')

    def test_after_promotion_fsync_failure(self):
        original=conv.promote
        def fail_fsync(*args):
            original(*args)
            patcher=patch('convert_fp16.os.fsync',side_effect=OSError('durability'))
            patcher.start();self.addCleanup(patcher.stop)
        with patch.object(conv,'promote',side_effect=fail_fsync):
            with self.assertRaises(aio.LifecycleError) as caught:self.run_conversion()
        self.assertTrue(caught.exception.state['promoted'])
        self.assertEqual(caught.exception.state['durability'],'uncertain')

    def test_multidimensional_scalar_and_torch_dtype(self):
        self.set_tensors([('matrix',[0x3f80]*4),('scalar',[0x4000])])
        rewrite_header(self.shard,lambda h:(h['matrix'].__setitem__('shape',[2,2]),h['scalar'].__setitem__('shape',[])))
        cfg=self.source/'config.json';data=json.loads(cfg.read_text());data['torch_dtype']='bfloat16';cfg.write_bytes(aio.canonical(data))
        self.anchor=source_manifest(self.source);self.run_conversion()
        out=json.loads((self.parent/'fp16'/'config.json').read_text())
        self.assertEqual(out['torch_dtype'],'float16');self.assertEqual(out['dtype'],'float16')
        data['torch_dtype']='float32';cfg.write_bytes(aio.canonical(data));self.anchor=source_manifest(self.source)
        with self.assertRaises(aio.AuditError):self.run_conversion('conflict')

# Do not rediscover imported original tests a second time.
del ConversionTests
