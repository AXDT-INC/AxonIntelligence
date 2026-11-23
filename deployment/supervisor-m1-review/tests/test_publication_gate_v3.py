"""Real native owner, actual conversion/promotion, synthetic fixtures only.
Raw frames deliberately bypass BOTH Python cancellation ledgers.
"""
import json,os,select,sys,time,unittest
from unittest.mock import patch
import candidate,native_engine as n,convert_fp16 as c,test_conversion as fixtures
from audit_io import LifecycleError

def raw_control(worker,op,payload=b''):
 s=worker.session;s.seq+=1;seq=s.seq;deadline=time.monotonic()+5
 def transfer(data=None,count=0):
  out=bytearray();view=memoryview(data) if data is not None else None
  while (len(view) if view is not None else count-len(out)):
   left=deadline-time.monotonic()
   if left<=0:raise RuntimeError('raw protocol deadline')
   rd,wr,_=select.select([3] if view is None else [],[4] if view is not None else [],[],left)
   if view is None:
    if not rd:continue
    b=os.read(3,count-len(out))
    if not b:raise RuntimeError('raw protocol EOF')
    out.extend(b)
   else:
    if not wr:continue
    size=os.write(4,view)
    if size<=0:raise RuntimeError('raw write failure')
    view=view[size:]
  return bytes(out)
 transfer(n.HEADER.pack(0x41585332,2,seq,op,worker.worker_id,len(payload))+payload)
 h=n.HEADER.unpack(transfer(count=24))
 if h[:5]!=(0x41585332,2,seq,op|0x80000000,worker.worker_id) or h[5]>131200:raise RuntimeError('raw response framing')
 return json.loads(transfer(count=h[5]))

class CommitGateV3(unittest.TestCase):
 def setUp(self):
  self.f=fixtures.ConversionTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
  self.state={};self.w=None;self.observed=[]
 def artifact(self):return self.f.parent/'commit-result'
 def convert(self):return c.convert(str(self.f.source),str(self.f.parent),'commit-result',self.f.anchor,0,self.state)
 def cancel(self,raw):
  before=self.artifact().exists()
  response=raw_control(self.w,4) if raw else self.w.control(4)
  self.observed.append({'artifact_at_ack':before,'response':response})
  return response
 def boundary(self,needle,raw,expected_artifact,completed):
  line=next(i for i,s in enumerate(open(c.__file__).read().splitlines(),1) if needle in s)
  def trace(frame,event,arg):
   if frame.f_code is c.convert.__code__ and event=='line' and frame.f_lineno==line and not self.observed:self.cancel(raw)
   return trace
  error=None
  try:
   with candidate.engine('native') as w:
    self.w=w;sys.settrace(trace);self.convert()
  except (n.WorkerError,LifecycleError) as e:error=e
  finally:sys.settrace(None)
  self.assertEqual(len(self.observed),1)
  response=self.observed[0]['response']
  self.assertIs(response['cancellation_requested'],True)
  self.assertIs(response['cancellation_complete'],completed)
  self.assertEqual(self.artifact().exists(),expected_artifact)
  self.assertIs(w.info['reaped'],True)
  if not expected_artifact:self.assertIs(self.state['promoted'],False);self.assertIsNotNone(error)
  else:self.assertIs(self.state['promoted'],True)
  # Completed cancellation preceded promotion only when commit was denied.
  if completed and not self.observed[0]['artifact_at_ack']:self.assertFalse(self.artifact().exists())
 def test_raw_cancel_after_finish_before_commit(self):
  self.boundary("__import__('native_engine').require_publication_authorized(state)",True,False,True)
 def test_direct_cancel_after_finish_before_commit(self):
  self.boundary("__import__('native_engine').require_publication_authorized(state)",False,False,True)
 def test_direct_cancel_after_grant_before_rename(self):
  self.boundary("state.update(phase='atomic_promotion'",False,True,False)
 def test_raw_cancel_after_grant_before_rename(self):
  self.boundary("state.update(phase='atomic_promotion'",True,True,False)
 def test_direct_cancel_after_rename_before_fsync(self):
  self.boundary("state.update(phase='post_promotion_fsync'",False,True,False)
 def test_raw_cancel_after_rename_before_fsync(self):
  self.boundary("state.update(phase='post_promotion_fsync'",True,True,False)
 def test_direct_cancel_after_durability_before_result(self):
  self.boundary("__import__('native_engine').complete_publication(state)",False,True,False)
 def test_raw_cancel_after_durability_before_result(self):
  self.boundary("__import__('native_engine').complete_publication(state)",True,True,False)
 def test_direct_cancel_after_confirmed_commit(self):
  self.boundary('return dict(result, **state)',False,True,True)
 def test_raw_cancel_after_confirmed_commit(self):
  self.boundary('return dict(result, **state)',True,True,True)
 def test_raw_cancel_status_cannot_erase_history(self):
  with self.assertRaises(n.WorkerError):
   with candidate.engine('native') as w:
    w.finish();a=raw_control(w,4);b=w.control(3)
    self.assertTrue(a['cancellation_complete']);self.assertTrue(b['cancellation_requested'])
    self.assertEqual(raw_control(w,7)['commit_decision'],'denied');w.finish()
  self.assertFalse(self.artifact().exists())
 def test_pending_cancel_repeated_until_commit_result(self):
  promote=c.promote
  def during(*a):
   first=self.cancel(True);second=self.cancel(True)
   self.assertFalse(first['cancellation_complete']);self.assertFalse(second['cancellation_complete'])
   self.assertFalse(raw_control(self.w,3)['cancellation_complete']);promote(*a)
  with self.assertRaises(n.WorkerError):
   with candidate.engine('native') as w:
    self.w=w
    with patch.object(c,'promote',during):self.convert()
    status=w.control(3);self.assertTrue(status['cancellation_complete']);self.assertTrue(self.artifact().exists())
  self.assertTrue(self.state['promoted'])
 def test_duplicate_commit_begin_denied_without_second_grant(self):
  promote=c.promote
  def during(*a):
   response=raw_control(self.w,7)
   self.assertEqual(response['commit_decision'],'denied');self.assertEqual(response['commit_state'],'in_progress')
   promote(*a)
  with candidate.engine('native') as w:
   self.w=w
   with patch.object(c,'promote',during):self.convert()
   self.assertEqual(w.info['commit_state'],'published')
  self.assertTrue(self.artifact().exists())
 def test_rename_failure_completes_pending_cancel_without_artifact(self):
  def fail(*a):self.cancel(True);raise OSError('synthetic rename failure')
  with self.assertRaises(LifecycleError):
   with candidate.engine('native') as w:
    self.w=w
    with patch.object(c,'promote',fail):self.convert()
  self.assertFalse(self.artifact().exists());self.assertFalse(self.state['promoted'])
  self.assertEqual(w.info['commit_state'],'not_published');self.assertTrue(w.info['cancellation_complete'])
 def test_unknown_rename_never_completes_pending_cancel(self):
  promote=c.promote
  def ambiguous(*a):self.cancel(True);promote(*a);raise KeyboardInterrupt('unknown rename return')
  with self.assertRaises(LifecycleError):
   with candidate.engine('native') as w:
    self.w=w
    with patch.object(c,'promote',ambiguous):self.convert()
  self.assertTrue(self.artifact().exists());self.assertIsNone(self.state['promoted'])
  self.assertEqual(w.info['commit_state'],'UNKNOWN');self.assertFalse(w.info['cancellation_complete'])
 def test_fsync_failure_preserves_publication_and_uncertain_durability(self):
  fsync=os.fsync
  def fail(fd):
   if self.state.get('promoted') is True:self.cancel(True);raise OSError('synthetic fsync')
   return fsync(fd)
  with self.assertRaises(LifecycleError):
   with candidate.engine('native') as w:
    self.w=w
    with patch.object(c.os,'fsync',fail):self.convert()
  self.assertTrue(self.artifact().exists());self.assertTrue(self.state['promoted'])
  self.assertEqual(self.state['durability'],'uncertain');self.assertEqual(w.info['commit_durability'],1)
  self.assertTrue(w.info['cancellation_complete'])
 def test_commit_result_reporting_error_preserves_primary(self):
  control=n.Native.control
  def fault(w,op,*a,**k):
   if op==8:raise SystemExit('secondary result failure')
   return control(w,op,*a,**k)
  def primary(*a):raise KeyboardInterrupt('primary rename interruption')
  with self.assertRaises(LifecycleError) as caught:
   with candidate.engine('native') as w:
    self.w=w
    with patch.object(n.Native,'control',fault),patch.object(c,'promote',primary):self.convert()
  self.assertIn('primary rename interruption',str(caught.exception));self.assertIsNone(self.state['promoted'])
  self.assertTrue(w.commit_result_attempted);self.assertFalse(w.commit_result_confirmed)
  self.assertTrue(any(e['operation']=='commit_result' for e in w.cleanup_errors))
  # Explicitly resolve the synthetic test's known no-rename result; no retry by production code.
  raw_control(w,8,n.struct.pack('<2I',0,0))
 def test_successful_commit_is_single_use(self):
  with candidate.engine('native') as w:
   self.w=w;self.convert();seq=w.session.seq;w.finish();w.finish();self.assertEqual(seq,w.session.seq)
   self.assertEqual(raw_control(w,7)['commit_decision'],'denied')
  self.assertTrue(self.artifact().exists())

 def test_direct_cancel_inside_promotion_operation(self):
  self.inside_promotion(False)
 def test_raw_cancel_inside_promotion_operation(self):
  self.inside_promotion(True)
 def inside_promotion(self,raw):
  # Inject inside the real promotion function, immediately before renameat2.
  line=next(i for i,s in enumerate(open(c.__file__).read().splitlines(),1) if 'if fn(parent.fd' in s)
  def trace(frame,event,arg):
   if event=='line' and frame.f_code is c.promote.__code__ and frame.f_lineno==line and not self.observed:
    response=self.cancel(raw)
    self.assertFalse(response['cancellation_complete']);self.assertFalse(self.artifact().exists())
   return trace
  with self.assertRaises(n.WorkerError):
   with candidate.engine('native') as w:
    self.w=w
    try:sys.settrace(trace);self.convert()
    finally:sys.settrace(None)
  self.assertTrue(self.artifact().exists());self.assertTrue(self.state['promoted'])
  self.assertTrue(w.info['cancellation_complete'])
