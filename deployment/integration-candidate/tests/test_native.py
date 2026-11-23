import os,sys,signal,struct,json,hashlib,tempfile,subprocess,unittest,fcntl,time
from pathlib import Path
from unittest.mock import patch
import test_conversion as fixtures
import convert_fp16 as c
import native_engine as n
import candidate
from audit_io import LifecycleError,canonical

class NativeFixtures(fixtures.ConversionTests):
 def run_conversion(self,name='fp16'):
  return candidate.convert(str(self.source),str(self.parent),name,self.anchor,selection='native')
 def test_scalar_matrix(self):
  self.set_tensors([('scalar',[0x3f80]),('matrix',[0x3f80]*6)])
  fixtures.rewrite_header(self.shard,lambda h:(h['scalar'].update(shape=[]),h['matrix'].update(shape=[2,3])))
  self.anchor=fixtures.source_manifest(self.source);self.run_conversion()
  self.assertEqual(candidate.verify(str(self.source),str(self.parent/'fp16'),self.anchor,selection='native')['status'],'PASS')
 def test_worker_signal_prevents_promotion_and_preserves_state(self):
  state={};original=n.Native.block
  def crash(worker,*args,**kwargs):
   os.kill(worker.proc.pid,signal.SIGKILL);return original(worker,*args,**kwargs)
  with self.assertRaises(n.WorkerError),candidate.engine('native'),patch.object(n.Native,'block',crash):
   with self.assertRaises(LifecycleError) as caught:c.convert(str(self.source),str(self.parent),'fp16',self.anchor,0,state)
  self.assertFalse(state['promoted']);self.assertEqual(state['phase'],'conversion');self.assertFalse((self.parent/'fp16').exists())
 def test_verifier_worker_failure_prevents_promotion(self):
  original=n.Native.block;state={}
  def crash(worker,raw,stats,actual=None):
   if actual is not None:os.kill(worker.proc.pid,signal.SIGTERM)
   return original(worker,raw,stats,actual)
  with self.assertRaises(n.WorkerError),candidate.engine('native'),patch.object(n.Native,'block',crash):
   with self.assertRaises(LifecycleError):c.convert(str(self.source),str(self.parent),'fp16',self.anchor,0,state)
  self.assertEqual(state['phase'],'numerical_verification');self.assertFalse(state['promoted'])
 def test_native_shutdown_failure_prevents_promotion(self):
  state={};finish=n.Native.finish
  def fail(w):os.kill(w.proc.pid,signal.SIGKILL);return finish(w)
  with self.assertRaises(Exception):
   with candidate.engine('native'),patch.object(n.Native,'finish',fail):c.convert(str(self.source),str(self.parent),'fp16',self.anchor,0,state)
  self.assertFalse(state['promoted']);self.assertFalse((self.parent/'fp16').exists())
 def test_interrupted_promotion_preserves_unknown(self):
  state={};promote=c.promote
  def interrupted(*args):promote(*args);raise KeyboardInterrupt('after rename')
  with candidate.engine('native'),patch.object(c,'promote',interrupted):
   with self.assertRaises(LifecycleError):c.convert(str(self.source),str(self.parent),'fp16',self.anchor,0,state)
  self.assertTrue((self.parent/'fp16').exists());self.assertIsNone(state['promoted']);self.assertEqual(state['durability'],'unknown')
 def test_cli_worker_failure_reporting(self):
  import io
  out=io.StringIO()
  class Receipt:
   def __init__(self,*a):pass
   def __enter__(self):return self
   def __exit__(self,*a):pass
  with self.assertRaises(n.WorkerError),candidate.engine('native') as worker,patch.object(c,'SOURCE',str(self.source)),patch.object(c,'CONVERTED',str(self.parent)),patch.object(c,'ReceiptDestination',Receipt),patch.object(sys,'argv',['tool','--artifact-name','fp16','--authorize-cpu-conversion','--receipt','synthetic']),patch.object(sys,'stdout',out):
   original=c.convert
   def work(*args,**kw):os.kill(worker.proc.pid,signal.SIGKILL);return original(*args,anchor=self.anchor,reserve_bytes=0,**kw)
   with patch.object(c,'convert',work):self.assertEqual(c.main(),1)
   # Context deliberately fails closed after a swallowed CLI failure.
   self.assertTrue(worker.broken)
  result=json.loads(out.getvalue());self.assertFalse(result['promoted']);self.assertEqual(result['phase'],'conversion')

 def test_actual_native_publication_survives_outer_cleanup_interrupt(self):
  import io
  from audit_io import Directory
  out=io.StringIO();record={};original=c.convert;close=Directory.__exit__
  class Receipt:
   def __init__(self,*args):pass
   def __enter__(self):return self
   def __exit__(self,*args):pass
   def publish(self,result):pass
  def work(*args,lifecycle):
   record['state']=lifecycle
   return original(*args,anchor=self.anchor,reserve_bytes=0,lifecycle=lifecycle)
  def interrupted(directory,*args):
   close(directory,*args)
   if record.get('state',{}).get('promoted') is True and not record.get('injected'):
    record['injected']=True;raise KeyboardInterrupt('post-promotion outer cleanup')
  with candidate.engine('native'),patch.object(c,'SOURCE',str(self.source)),patch.object(c,'CONVERTED',str(self.parent)),patch.object(c,'ReceiptDestination',Receipt),patch.object(c,'convert',work),patch.object(Directory,'__exit__',interrupted),patch.object(sys,'argv',['tool','--artifact-name','fp16','--authorize-cpu-conversion','--receipt','synthetic']),patch.object(sys,'stdout',out):
   self.assertEqual(c.main(),1)
  state=json.loads(out.getvalue());self.assertTrue(state['promoted']);self.assertEqual(state['durability'],'confirmed');self.assertEqual(state['phase'],'conversion_cleanup');self.assertTrue((self.parent/'fp16').exists())

 def test_rehashed_native_provenance_tampering_rejected(self):
  self.run_conversion();output=self.parent/'fp16';receipt=output/'CONVERSION.json'
  data=json.loads(receipt.read_text());data['candidate_engine']['executable_sha256']='0'*64;receipt.write_bytes(canonical(data));fixtures.update_output_manifest(output)
  with self.assertRaisesRegex(n.WorkerError,'provenance'):candidate.verify(str(self.source),str(output),self.anchor,selection='reference')

class Protocol(unittest.TestCase):
 def test_block_split_exact_order_and_statistics(self):
  raw=struct.pack('<'+'H'*65536,*([0x3300,0x3301,0x8000,0x3f80]*16384))
  st=c.empty_stats();expected=bytes(c.reference_convert_block(raw,st))
  with candidate.engine('native') as w:
   actual=c.empty_stats();self.assertEqual(w.block(raw,actual),expected);self.assertEqual(actual,st)
   verified=c.empty_stats();w.block(raw,verified,expected);self.assertEqual(verified,st)
 def test_all_encodings_native_sources_and_v3(self):
  # Fresh standalone test binary exercises the same unmodified numerical C sources.
  root=n.ROOT
  with tempfile.TemporaryDirectory(dir=root/'test-workspace') as td:
   t=Path(td);(t/'all.c').write_text('''#include "prototype.h"\n#include <stdio.h>\nint main(void){for(unsigned b=0;b<65536;b++){unsigned char in[2]={b,b>>8},out[2]={0};Statistics s={0};unsigned char rc=prototype_convert(in,2,out,2,&s);if(!rc){Statistics v={0};if(prototype_verify(in,2,out,2,&v))return 2;}fwrite(&rc,1,1,stdout);fwrite(out,1,2,stdout);fwrite(&s,1,80,stdout);}return 0;}''')
   subprocess.run(['/usr/bin/gcc','-O2','-fno-fast-math','-ffp-contract=off','-I',str(root),str(t/'all.c'),str(root/'convert.c'),str(root/'verify.c'),'-lm','-o',str(t/'all')],env={'PATH':'/usr/bin:/bin','TMPDIR':td},check=True)
   data=subprocess.check_output([str(t/'all')]);self.assertEqual(len(data),65536*83)
   for b in range(65536):
    record=data[b*83:(b+1)*83];st=c.empty_stats()
    try:out=bytes(c.reference_convert_block(struct.pack('<H',b),st))
    except Exception:self.assertEqual(record[0],1 if b&0x7f80==0x7f80 else 2)
    else:self.assertEqual(record[0],0);self.assertEqual(record[1:3],out);self.assertEqual(dict(zip(n.KEYS,n.STATS.unpack(record[3:]))),st)
 def test_timeout_no_retry(self):
  w=n.Native(timeout=.1)
  try:
   os.kill(w.proc.pid,signal.SIGSTOP)
   with self.assertRaisesRegex(n.WorkerError,'timeout'):w.block(b'\0\0',c.empty_stats())
   with self.assertRaises(n.WorkerError):w.block(b'\0\0',c.empty_stats())
  finally:w.abort()
 def test_malformed_response_no_retry(self):
  w=n.Native()
  try:
   with patch.object(w,'_read',return_value=b'\0'*24):
    with self.assertRaisesRegex(n.WorkerError,'malformed'):w.block(b'\0\0',c.empty_stats())
   self.assertTrue(w.closed)
  finally:w.abort()
 def test_interruption_kills_worker(self):
  w=n.Native()
  try:
   with patch.object(w,'_read',side_effect=KeyboardInterrupt):
    with self.assertRaises(KeyboardInterrupt):w.block(b'\0\0',c.empty_stats())
   self.assertIsNotNone(w.proc.poll())
  finally:w.abort()
 def test_authentication_symlink_corruption_and_seals(self):
  import shutil
  with tempfile.TemporaryDirectory(dir=n.ROOT/'test-workspace') as td:
   root=Path(td)
   for name in ['trust.json','worker','worker.c','convert.c','verify.c','prototype.h']:shutil.copyfile(n.ROOT/name,root/name);(root/name).chmod(0o600)
   with patch.object(n,'ROOT',root):
    fd,trust=n.authenticated_image()
    try:
     with self.assertRaises(OSError):os.write(fd,b'corrupt')
     # Path replacement cannot change authenticated sealed bytes.
     (root/'worker').unlink();(root/'worker').write_bytes(b'fake')
     os.lseek(fd,0,0);self.assertEqual(hashlib.sha256(os.read(fd,4*1024**2)).hexdigest(),trust['executable_sha256'])
     with patch.object(n,'authenticated_image',return_value=(os.dup(fd),trust)):
      worker=n.Native();self.assertEqual(worker.block(b'\x80\x3f',c.empty_stats()),b'\0\x3c');worker.finish()
     with self.assertRaises(n.WorkerError):n.authenticated_image()
     (root/'worker').unlink();(root/'worker').symlink_to(n.ROOT/'worker.c')
     with self.assertRaises(OSError):n.authenticated_image()
    finally:os.close(fd)

class Sandbox(unittest.TestCase):
 def test_filesystem_syscall_killed(self):
  root=n.ROOT
  with tempfile.TemporaryDirectory(dir=root/'test-workspace') as td:
   t=Path(td)
   (t/'probe.c').write_text('#define main original_worker_main\n#include "worker.c"\n#undef main\nint main(void){sandbox();syscall(__NR_openat,-100,"forbidden-file",0,0);return 99;}\n')
   subprocess.run(['/usr/bin/gcc','-O2','-I',str(root),str(t/'probe.c'),str(root/'convert.c'),str(root/'verify.c'),'-lm','-Wl,-z,now','-o',str(t/'probe')],env={'PATH':'/usr/bin:/bin','TMPDIR':td},check=True)
   result=subprocess.run([str(t/'probe')],cwd=t)
   self.assertEqual(result.returncode,-signal.SIGSYS);self.assertFalse((t/'forbidden-file').exists())

class Additional(unittest.TestCase):
 def test_all_valid_patterns_through_protocol(self):
  raw=b''.join(struct.pack('<H',b) for b in range(65536) if (b&0x7fff)<=0x477f)
  ref=c.empty_stats();expected=bytes(c.reference_convert_block(raw,ref))
  with candidate.engine('native') as worker:
   st=c.empty_stats();actual=worker.block(raw,st);self.assertEqual(actual,expected);self.assertEqual(st,ref)
   st=c.empty_stats();worker.block(raw,st,actual);self.assertEqual(st,ref)
 def test_startup_failure_is_preflight_lifecycle(self):
  with patch.object(n,'authenticated_image',side_effect=n.WorkerError('authentication')):
   with self.assertRaises(LifecycleError) as caught:candidate.convert('unused','unused','unused','unused',selection='native')
  self.assertFalse(caught.exception.state['promoted']);self.assertEqual(caught.exception.state['phase'],'preflight')
