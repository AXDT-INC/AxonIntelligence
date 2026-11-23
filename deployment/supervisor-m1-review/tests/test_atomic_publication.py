import ctypes,os,threading,unittest
from unittest.mock import patch
import candidate,native_engine as n,convert_fp16 as c,test_conversion as fixtures
from audit_io import LifecycleError
class AtomicPublication(unittest.TestCase):
 def setUp(self):
  self.f=fixtures.ConversionTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
 def library(self,w):
  # Same authenticated library, releasing the GIL only for C concurrency tests.
  lib=ctypes.CDLL(None,handle=w.publication_library._handle,use_errno=True)
  for name in ('gate_create','gate_arm','gate_invalidate','gate_status','publish_once'):
   a=getattr(w.publication_library,name);b=getattr(lib,name);b.argtypes=a.argtypes;b.restype=a.restype
  return lib
 def run_order(self,order):
  with candidate.engine('native') as w:
   lib=self.library(w)
   for i in range(8):
    g=lib.gate_create(os.getpid(),1000000+w.worker_id*100+i);self.assertTrue(g);self.assertFalse(lib.gate_create(os.getpid(),1000000+w.worker_id*100+i));self.assertEqual(lib.gate_arm(g,os.getpid(),1000000+w.worker_id*100+i),0)
    source=self.f.parent/('stage'+str(i));source.mkdir();dest=self.f.parent/('out'+str(i));fd=os.open(self.f.parent,os.O_RDONLY|os.O_DIRECTORY)
    barrier=threading.Barrier(2);done=threading.Event();results={};errors=[]
    def task(kind):
     try:
      barrier.wait(timeout=5)
      if order!='concurrent' and kind!=order:
       if not done.wait(5):raise RuntimeError('ordering event timeout')
      if kind=='invalidate':results[kind]=lib.gate_invalidate(g,os.getpid(),1000000+w.worker_id*100+i)
      else:results[kind]=lib.publish_once(fd,g,os.getpid(),1000000+w.worker_id*100+i,source.name.encode(),dest.name.encode())
      if kind==order:done.set()
     except BaseException as e:errors.append(repr(e));done.set()
    threads=[threading.Thread(target=task,args=(kind,)) for kind in ('invalidate','publish')]
    try:
     for thread in threads:thread.start()
     for thread in threads:thread.join(10)
     self.assertFalse(any(thread.is_alive() for thread in threads));self.assertEqual(errors,[])
     out=(ctypes.c_uint64*7)();self.assertEqual(lib.gate_status(g,os.getpid(),1000000+w.worker_id*100+i,out),0)
     self.assertEqual(results['invalidate'],0);self.assertEqual(dest.exists(),results['publish']==0)
     self.assertEqual(out[3],1);self.assertEqual(bool(out[5]),dest.exists())
     if order=='invalidate':self.assertFalse(dest.exists());self.assertEqual(out[4],0)
     if order=='publish':self.assertTrue(dest.exists());self.assertEqual(out[4],1)
     self.assertEqual(lib.gate_arm(g,os.getpid(),1000000+w.worker_id*100+i),-1)
    finally:os.close(fd)
 def test_invalidation_first_prevents_syscall(self):self.run_order('invalidate')
 def test_promotion_first_preserves_confirmation(self):self.run_order('publish')
 def test_concurrent_requests_have_one_serialized_outcome(self):self.run_order('concurrent')
 def test_wrong_identity_does_not_invalidate_another_session(self):
  with candidate.engine('native') as w:
   lib=self.library(w);g=lib.gate_create(os.getpid(),90001);self.assertEqual(lib.gate_arm(g,os.getpid(),90001),0)
   self.assertEqual(lib.gate_invalidate(g,os.getpid(),90002),-1)
   out=(ctypes.c_uint64*7)();self.assertEqual(lib.gate_status(g,os.getpid(),90001,out),0);self.assertEqual(out[2:4],[1,0])
   self.assertEqual(lib.gate_invalidate(g,os.getpid(),90001),0);self.assertEqual(lib.gate_invalidate(g,os.getpid(),90001),0)
 def test_interruption_after_native_return_retains_native_confirmation(self):
  original=c.promote;state={}
  def interrupt(*args):original(*args);raise KeyboardInterrupt('after native syscall')
  with self.assertRaises(LifecycleError):
   with candidate.engine('native') as w:
    with patch.object(c,'promote',interrupt):c.convert(str(self.f.source),str(self.f.parent),'interrupted',self.f.anchor,0,state)
  self.assertTrue((self.f.parent/'interrupted').exists());self.assertIsNone(state['promoted'])
  native=w.native_publication_status();self.assertEqual(native['confirmed'],1);self.assertEqual(native['attempted'],1)
 def test_native_failure_is_recorded_and_cannot_retry(self):
  with candidate.engine('native') as w:
   lib=self.library(w);g=lib.gate_create(os.getpid(),90003);self.assertEqual(lib.gate_arm(g,os.getpid(),90003),0)
   fd=os.open(self.f.parent,os.O_RDONLY|os.O_DIRECTORY)
   try:
    self.assertEqual(lib.publish_once(fd,g,os.getpid(),90003,b'absent',b'new'),-1)
    (self.f.parent/'absent').mkdir()
    self.assertEqual(lib.publish_once(fd,g,os.getpid(),90003,b'absent',b'new'),-1)
    self.assertFalse((self.f.parent/'new').exists())
    out=(ctypes.c_uint64*7)();lib.gate_status(g,os.getpid(),90003,out);self.assertEqual(out[4],1);self.assertEqual(out[5],0);self.assertNotEqual(out[6],0)
   finally:os.close(fd)
