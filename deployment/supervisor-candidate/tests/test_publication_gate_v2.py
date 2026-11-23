"""Publication authorization tests: real supervised workers and small fixtures.
Response mutations occur only after the actual supervisor processed FINISH.
Thus the native harness, not mocked process objects, establishes real reaping.
"""
import io,json,os,signal,sys,unittest
from unittest.mock import patch
import candidate,native_engine as n,convert_fp16 as c
import test_conversion as fixtures
from audit_io import LifecycleError

class PublicationGate(unittest.TestCase):
 def setUp(self):
  self.fixture=fixtures.ConversionTests();self.fixture.setUp()
  self.addCleanup(self.fixture.doCleanups)
  self.state={};self.worker=None
 def convert(self):
  f=self.fixture
  return c.convert(str(f.source),str(f.parent),'gate-result',f.anchor,0,self.state)
 def artifact_exists(self):return (self.fixture.parent/'gate-result').exists()
 def assert_reaped(self,worker):
  # Confirmation comes from the real native owner; absence alone is insufficient.
  self.assertIs(worker.info['reaped'],True)
  self.assertEqual(worker.info['process_state'],'exited_reaped')
  with self.assertRaises(ProcessLookupError):os.kill(worker.worker_pid,0)
 def denied_at_hook(self,inject):
  original=n.finish_before_publication
  def hook(result,lifecycle):
   self.assertEqual(result['status'],'PASS')
   self.assertEqual(lifecycle['numerical_verification'],'PASS')
   inject(self.worker)
   return original(result,lifecycle)
  with self.assertRaises(LifecycleError):
   with candidate.engine('native') as w:
    self.worker=w
    with patch.object(n,'finish_before_publication',hook):self.convert()
  self.assertFalse(self.artifact_exists())
  self.assertIs(self.state['promoted'],False)
  self.assertFalse(w.snapshot()['publication_authorized'])
  self.assert_reaped(w)
  return w
 def test_direct_cancel_after_verified_before_publication(self):
  w=self.denied_at_hook(lambda w:w.control(4))
  self.assertTrue(w.cancellation_occurred)
  self.assertTrue(w.exit_facts()['worker_exited_unsuccessfully'])
  self.assertEqual(w.exit_facts()['worker_exit_signal'],signal.SIGKILL)
  self.assertFalse(w.finish_acknowledged)
 def test_raw_session_cancel_is_also_terminal(self):
  w=self.denied_at_hook(lambda w:w.session.exchange(4,w.worker_id))
  self.assertTrue(w.cancellation_occurred)
 def test_repeated_cancel_and_successful_cleanup_do_not_restore_eligibility(self):
  def cancel(w):
   for _ in range(3):w.control(4)
   self.assertTrue(w.closed)
  w=self.denied_at_hook(cancel)
  for _ in range(2):
   w.abort()
   with self.assertRaises(n.WorkerError):w.finish()
  self.assertTrue(w.closed);self.assertTrue(w.cancellation_occurred)
 def test_cancel_after_successful_finish_before_atomic_promotion(self):
  fsync=c.os.fsync;cancelled=[]
  def interrupt(fd):
   fsync(fd)
   if self.worker.finish_validated and not cancelled:
    cancelled.append(True);self.worker.control(4)
  with self.assertRaises(LifecycleError):
   with candidate.engine('native') as w:
    self.worker=w
    with patch.object(c.os,'fsync',interrupt):self.convert()
  self.assertTrue(cancelled);self.assertFalse(self.artifact_exists())
  self.assertIs(self.state['promoted'],False);self.assertTrue(w.finish_validated)
  self.assertFalse(w.snapshot()['publication_authorized']);self.assert_reaped(w)
 def test_hook_cannot_be_replaced_by_cleanup_or_plain_finish(self):
  with self.assertRaises(LifecycleError):
   with candidate.engine('native') as w:
    # Even a successful normal shutdown is insufficient without the full
    # numerical-verification authorization and the boundary recheck.
    with patch.object(n,'finish_before_publication',lambda *a:w.finish()):self.convert()
  self.assertFalse(self.artifact_exists());self.assertIs(self.state['promoted'],False)
  self.assertTrue(w.finish_validated);self.assertFalse(w.numerical_work_completed)
  self.assert_reaped(w)
 def test_direct_finish_without_verified_result_does_not_authorize_publication(self):
  with self.assertRaises(n.WorkerError),candidate.engine('native') as w:
   w.control(6);w.finish()
   self.assertTrue(w.finish_validated)
   with self.assertRaises(n.WorkerError):w.require_publication_authorized(self.state)
   # Failed authorization intentionally makes this context fail closed.
   self.assertFalse(w.snapshot()['publication_authorized'])
 def response_failure(self,mutation=None,error=None):
  exchange=n.SESSION.exchange;observed=[]
  def altered(op,token,*args,**kwargs):
   raw=exchange(op,token,*args,**kwargs)
   if op==6:
    response=json.loads(raw);observed.append(dict(response))
    if error:raise error
    mutation(response)
    return json.dumps(response).encode()
   return raw
  with patch.object(n.SESSION,'exchange',altered):
   w=self.denied_at_hook(lambda w:None)
  self.assertTrue(observed);self.assertTrue(observed[0]['reaped'])
  self.assertEqual(observed[0]['wait_status'],0)
  return w
 def test_nonzero_exit_status_denies_publication(self):
  w=self.response_failure(lambda r:r.update(wait_status=7<<8))
  self.assertTrue(w.cancellation_occurred);self.assertFalse(w.finish_validated)
  self.assertEqual(w.finish_response['wait_status'],7<<8)
  w.control(3) # later real STATUS reports clean reaping, but cannot repair FINISH
  self.assertEqual(w.info['wait_status'],0)
  with self.assertRaises(n.WorkerError):w.finish()
 def test_signal_wait_status_denies_publication(self):
  self.response_failure(lambda r:r.update(wait_status=signal.SIGKILL))
 def test_missing_finish_ack_denies_publication(self):
  w=self.response_failure(error=n.WorkerError('missing FINISH acknowledgment'))
  self.assertFalse(w.finish_acknowledged)
 def test_invalid_finish_token_denies_publication(self):
  self.response_failure(lambda r:r.update(token=r['token']+1))
 def test_invalid_finish_pid_denies_publication(self):
  self.response_failure(lambda r:r.update(pid=r['pid']+1))
 def test_acknowledged_finish_without_reaping_denies_publication(self):
  self.response_failure(lambda r:r.update(reaped=False,cleanup_complete=False,process_state='cleanup_incomplete'))
 def test_acknowledged_finish_with_cleanup_error_denies_publication(self):
  self.response_failure(lambda r:r.update(cleanup_errors=[1005],error_count=1))
 def test_truncated_cleanup_diagnostics_deny_publication(self):
  self.response_failure(lambda r:r.update(errors_truncated_count=1))
 def test_unknown_wait_status_denies_publication(self):
  self.response_failure(lambda r:r.update(wait_status=None))
 def test_normal_exit_and_signal_decoding_are_distinct(self):
  with candidate.engine('native') as w:
   w.finish();original=dict(w.info)
   try:
    for status,normal,code,sig in [(0,True,0,None),(7<<8,True,7,None),(9,False,None,9),(127,False,None,None)]:
     w.info=dict(original,wait_status=status)
     facts=w.exit_facts();self.assertEqual((facts['worker_exited_normally'],facts['worker_exit_code'],facts['worker_exit_signal']),(normal,code,sig))
   finally:w.info=original
 def test_successful_finish_is_idempotent_without_retransmission(self):
  with candidate.engine('native') as w:
   self.worker=w;self.convert();sequence=w.session.seq
   for _ in range(3):w.finish()
   self.assertEqual(sequence,w.session.seq)
   self.assertTrue(w.snapshot()['publication_already_confirmed'])
  self.assertTrue(self.artifact_exists());self.assertIs(self.state['promoted'],True)
  self.assert_reaped(w)
 def test_cancel_during_context_exit_preserves_confirmed_publication(self):
  finish=n.Native.finish
  def cancelled(w):
   if self.state.get('promoted') is True:w.control(4)
   finish(w)
  with self.assertRaises(n.WorkerError),patch.object(n.Native,'finish',cancelled):
   with candidate.engine('native') as w:
    self.worker=w;self.convert()
  self.assertTrue(self.artifact_exists());self.assertIs(self.state['promoted'],True)
  self.assertTrue(w.snapshot()['publication_already_confirmed'])
  self.assertFalse(w.snapshot()['publication_authorized']);self.assert_reaped(w)
 def test_cleanup_error_after_publication_keeps_publication_fact(self):
  with self.assertRaises(n.WorkerError):
   with candidate.engine('native') as w:
    self.worker=w;self.convert();w.diagnostic('context_cleanup',OSError('secondary'))
  self.assertTrue(self.artifact_exists());self.assertIs(self.state['promoted'],True)
  self.assertEqual(self.state['durability'],'confirmed')
  self.assertTrue(w.snapshot()['publication_already_confirmed']);self.assert_reaped(w)
 def test_unknown_promotion_outcome_survives_cancellation(self):
  promote=c.promote
  def uncertain(*args):
   promote(*args);self.worker.control(4);raise KeyboardInterrupt('rename outcome not recorded')
  with self.assertRaises(LifecycleError):
   with candidate.engine('native') as w:
    self.worker=w
    with patch.object(c,'promote',uncertain):self.convert()
  self.assertTrue(self.artifact_exists());self.assertIsNone(self.state['promoted'])
  self.assertIsNone(w.snapshot()['publication_outcome']);self.assertFalse(w.snapshot()['publication_already_confirmed'])
  self.assert_reaped(w)
 def test_cancellation_during_interrupted_reporting_preserves_published_state(self):
  f=self.fixture;original=c.convert;out=io.StringIO();err=io.StringIO();worker=[]
  class Receipt:
   def __init__(self,*args):pass
   def __enter__(self):return self
   def __exit__(self,*args):pass
   def publish(self,result):raise KeyboardInterrupt('receipt publication interrupted')
  def convert(*args,**kwargs):
   self.state=kwargs['lifecycle']
   return original(*args,anchor=f.anchor,reserve_bytes=0,**kwargs)
  def reporting(record):
   worker[0].control(4);raise SystemExit('report interrupted')
  with self.assertRaises(n.WorkerError):
   with candidate.engine('native') as w:
    worker.append(w)
    with patch.object(c,'SOURCE',str(f.source)),patch.object(c,'CONVERTED',str(f.parent)),patch.object(c,'convert',convert),patch.object(c,'ReceiptDestination',Receipt),patch.object(c,'emit_receipt',reporting),patch.object(sys,'argv',['tool','--artifact-name','gate-result','--authorize-cpu-conversion','--receipt','synthetic']),patch.object(sys,'stdout',out),patch.object(sys,'stderr',err):
     self.assertEqual(c.main(),1)
  report=json.loads(err.getvalue());self.assertIs(report['promoted'],True)
  self.assertEqual(report['error'],'receipt publication interrupted')
  self.assertEqual(report['reporting_error_type'],'SystemExit')
  self.assertTrue(self.artifact_exists());self.assertTrue(w.snapshot()['publication_already_confirmed'])
  self.assert_reaped(w)

 def test_prepublication_filesystem_failure_cannot_reuse_authorization(self):
  fsync=c.os.fsync
  def fail(fd):
   if self.worker.finish_validated:raise OSError('stage fsync failed')
   return fsync(fd)
  with self.assertRaises(n.WorkerError):
   with candidate.engine('native') as w:
    self.worker=w
    with patch.object(c.os,'fsync',fail):
     with self.assertRaises(LifecycleError):self.convert()
    self.assertTrue(w.finish_validated)
    self.assertFalse(w.snapshot()['publication_authorized'])
    with self.assertRaises(n.WorkerError):w.require_publication_authorized(self.state)
  self.assertFalse(self.artifact_exists());self.assertIs(self.state['promoted'],False)
  self.assert_reaped(w)

 def test_session_loss_after_finish_denies_publication_without_forgetting_reap(self):
  # Session failure is injected only in the proxy's knowledge state, after
  # real FINISH/reaping, and restored for the test runner's next fresh worker.
  original=n.SESSION.failed
  with self.assertRaises(n.WorkerError):
   with candidate.engine('native') as w:
    w.finish()
    try:
     n.SESSION.failed=True
     with self.assertRaises(n.WorkerError):w.finish()
     self.assertTrue(w.info['reaped']);self.assertFalse(w.snapshot()['publication_authorized'])
     self.assertTrue(w.cancellation_occurred)
    finally:n.SESSION.failed=original
  self.assertTrue(w.cancellation_occurred);self.assert_reaped(w)
