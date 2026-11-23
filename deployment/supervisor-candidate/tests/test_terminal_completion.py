import sys,json,unittest
from unittest.mock import patch
import candidate,native_engine as n,convert_fp16 as c,test_conversion as fixtures
from test_publication_gate_v3 import raw_control
from audit_io import LifecycleError
class TerminalCompletion(unittest.TestCase):
 def setUp(self):
  self.f=fixtures.ConversionTests();self.f.setUp();self.addCleanup(self.f.doCleanups);self.state={};self.w=None
 def convert(self):return c.convert(str(self.f.source),str(self.f.parent),'terminal-result',self.f.anchor,0,self.state)
 def exists(self):return (self.f.parent/'terminal-result').exists()
 def premature(self,swallow):
  observed=[];line=next(i for i,s in enumerate(open(c.__file__).read().splitlines(),1) if "state.update(phase='atomic_promotion'" in s)
  def trace(frame,event,arg):
   if event=='line' and frame.f_code is c.convert.__code__ and frame.f_lineno==line and not observed:
    observed.append(True)
    try:n.complete_publication(self.state)
    except n.WorkerError:
     self.assertFalse(self.w.commit_result_attempted)
     reply=raw_control(self.w,4);self.assertFalse(reply['cancellation_complete'])
     self.assertEqual(reply['commit_state'],'in_progress');self.assertFalse(self.exists())
     if not swallow:raise
    else:self.fail('premature completion accepted')
   return trace
  with self.assertRaises(LifecycleError):
   with candidate.engine('native') as w:
    self.w=w
    try:sys.settrace(trace);self.convert()
    finally:sys.settrace(None)
  self.assertTrue(observed);self.assertFalse(self.exists());self.assertTrue(w.info['reaped'])
 def test_premature_completion_then_cancel(self):self.premature(False)
 def test_swallowed_premature_completion_cannot_resume_promotion(self):self.premature(True)
 def test_identical_completion_idempotent_and_promotion_consumed(self):
  with candidate.engine('native') as w:
   self.w=w;self.convert();seq=w.session.seq
   w.complete_commit(self.state);w.complete_commit(self.state);self.assertEqual(seq,w.session.seq)
   self.assertEqual(w.coordinator_phase,'commit_result_finalized')
   with self.assertRaises(n.WorkerError):w.start_promotion(self.state)
  self.assertTrue(self.exists())
 def test_conflicting_completion_rejected(self):
  with self.assertRaises(n.WorkerError):
   with candidate.engine('native') as w:
    self.w=w;self.convert();seq=w.session.seq
    self.state['promoted']=False
    try:
     with self.assertRaisesRegex(n.WorkerError,'conflicting'):w.complete_commit(self.state)
    finally:self.state['promoted']=True
    self.assertEqual(seq,w.session.seq)
  self.assertTrue(self.exists());self.assertTrue(self.state['promoted'])
 def test_different_lifecycle_rejected(self):
  with self.assertRaises(n.WorkerError):
   with candidate.engine('native') as w:
    self.w=w;self.convert()
    with self.assertRaisesRegex(n.WorkerError,'different'):w.complete_commit(dict(self.state))
  self.assertTrue(self.exists())
 def test_raw_forgery_remains_trusted_channel_limitation(self):
  original=c.promote;observed=[]
  def forged(*args):
   reply=raw_control(self.w,8,n.struct.pack('<2I',0,0));self.assertEqual(reply['commit_state'],'not_published')
   cancel=raw_control(self.w,4);self.assertTrue(cancel['cancellation_complete']);observed.append(self.exists())
   return original(*args)
  with self.assertRaises(LifecycleError):
   with candidate.engine('native') as w:
    self.w=w
    with patch.object(c,'promote',forged):self.convert()
  self.assertEqual(observed,[False]);self.assertTrue(self.exists());self.assertTrue(self.state['promoted'])
  self.assertFalse(w.commit_result_confirmed)
