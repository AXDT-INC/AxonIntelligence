import os,fcntl,tempfile,unittest,hashlib
from pathlib import Path
import bootstrap
class Trust(unittest.TestCase):
 def test_sealed_bytes_survive_path_replacement(self):
  with tempfile.TemporaryDirectory(dir=bootstrap.ROOT/'test-workspace') as td:
   p=Path(td)/'image';p.write_bytes(b'authenticated synthetic bytes');p.chmod(0o600)
   d=os.open(td,os.O_RDONLY|os.O_DIRECTORY)
   try:raw=bootstrap.read(d,'image',100)
   finally:os.close(d)
   fd=bootstrap.image(raw,'synthetic-seal')
   try:
    self.assertEqual(fcntl.fcntl(fd,fcntl.F_GET_SEALS),bootstrap.SEALS)
    with self.assertRaises(OSError):os.write(fd,b'bad')
    p.unlink();p.write_bytes(b'replaced');os.lseek(fd,0,0)
    self.assertEqual(os.read(fd,100),raw)
   finally:os.close(fd)
 def test_trusted_read_rejects_aliases_and_unsafe_targets(self):
  with tempfile.TemporaryDirectory(dir=bootstrap.ROOT/'test-workspace') as td:
   p=Path(td);(p/'file').write_bytes(b'abc');(p/'file').chmod(0o600);(p/'link').symlink_to(p/'file');os.link(p/'file',p/'hard')
   d=os.open(td,os.O_RDONLY|os.O_DIRECTORY)
   before={int(x) for x in os.listdir('/proc/self/fd')}
   try:
    for name in ('link','hard','../file','/file'):
     with self.assertRaises((ValueError,OSError)):bootstrap.read(d,name,100)
    (p/'hard').unlink()
    with self.assertRaises(ValueError):bootstrap.read(d,'file',1)
    (p/'file').chmod(0o666)
    with self.assertRaises(ValueError):bootstrap.read(d,'file',100)
   finally:
    os.close(d)
    # Bootstrap failures normally exit the whole childless process. Here the
    # test catches them, so close only descriptors allocated by these calls.
    for name in os.listdir('/proc/self/fd'):
     fd=int(name)
     if fd not in before:
      try:os.close(fd)
      except OSError:pass

class ReportingCleanup(unittest.TestCase):
 def test_channel_cleanup_accumulates_baseexceptions_without_masking(self):
  import native_engine as n
  from unittest.mock import patch
  session=n.Session()
  with patch.object(n.os,'close',side_effect=SystemExit('secondary')):
   session.close();session.close()
  self.assertEqual(len(session.errors),4)
  self.assertEqual(set(session.descriptors),{3,4})
  self.assertTrue(all(e['error_type']=='SystemExit' for e in session.errors))
  # This test-local ledger does not own or close the live coordinator channel.
 def test_replaced_descriptor_not_closed(self):
  import native_engine as n
  from unittest.mock import patch
  session=n.Session();session.descriptors={3:(-1,-1)}
  with patch.object(n.os,'close') as close:session.close();close.assert_not_called()
  self.assertFalse(session.descriptors);self.assertEqual(len(session.errors),1)
