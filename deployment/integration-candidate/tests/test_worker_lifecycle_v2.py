"""Real children plus targeted faults; teardown independently kills/reaps every child."""
import os,sys,signal,subprocess,unittest,io,json
from unittest.mock import patch
from contextlib import ExitStack
import native_engine as n,candidate,convert_fp16 as c
from audit_io import LifecycleError
import test_conversion as fixtures

OBSERVED_PIDS=[]
class Ownership(unittest.TestCase):
 def setUp(self):
  self.children=[];self.original_init=subprocess.Popen.__init__
  original=self.original_init
  def tracked(proc,*args,**kwargs):
   self.children.append(proc)
   result=original(proc,*args,**kwargs)
   if getattr(proc,'pid',None) is not None:OBSERVED_PIDS.append(proc.pid)
   return result
  self.tracker=patch.object(subprocess.Popen,'__init__',tracked);self.tracker.start()
 def tearDown(self):
  self.tracker.stop()
  for worker in list(n.OWNED.values()):
   worker.context_pending=False;worker.abort()
  n.ACTIVE.set(None)
  for proc in self.children:
   if getattr(proc,'pid',None) is not None:
    if proc.poll() is None:proc.kill()
    proc.wait(timeout=2)
    for stream in (proc.stdin,proc.stdout,proc.stderr):
     if stream is not None:stream.close()
    with self.assertRaises(ProcessLookupError):os.kill(proc.pid,0)
  self.assertFalse(n.OWNED)
 def assert_reaped(self):
  self.assertTrue(self.children)
  for proc in self.children:
   if getattr(proc,'pid',None) is not None:self.assertIsNotNone(proc.poll())
 def test_interrupt_before_popen_closes_image(self):
  before=set(os.listdir('/proc/self/fd'))
  for exc in (KeyboardInterrupt('before'),SystemExit(2)):
   with patch.object(subprocess.Popen,'__init__',side_effect=exc):
    with self.assertRaises(type(exc)) as caught:n.Native()
   self.assertEqual(caught.exception.worker_lifecycle['process_state'],'not_launched')
   self.assertTrue(caught.exception.worker_lifecycle['cleanup_complete'])
  self.assertEqual(set(os.listdir('/proc/self/fd')),before)
 def test_interrupt_after_actual_popen(self):
  original=subprocess.Popen.__init__
  for exc in (KeyboardInterrupt('after'),SystemExit(3)):
   def launch(proc,*args,**kwargs):original(proc,*args,**kwargs);raise exc
   with patch.object(subprocess.Popen,'__init__',launch):
    with self.assertRaises(type(exc)) as caught:n.Native()
   self.assertIs(caught.exception,exc);self.assertTrue(exc.worker_lifecycle['reaped']);self.assertTrue(exc.worker_lifecycle['cleanup_complete'])
  self.assert_reaped()
 def test_real_sigint_during_launch_is_deferred_until_owned(self):
  original=subprocess.Popen.__init__
  def launch(proc,*args,**kwargs):
   original(proc,*args,**kwargs);os.kill(os.getpid(),signal.SIGINT)
  with patch.object(subprocess.Popen,'__init__',launch):
   with self.assertRaises(KeyboardInterrupt):n.Native()
  self.assert_reaped()
 def test_memfd_close_interruption_after_launch(self):
  close=os.close
  def interrupt(fd):
   link=os.readlink('/proc/self/fd/'+str(fd))
   close(fd)
   if 'memfd:axon-candidate' in link:raise KeyboardInterrupt('image close after close')
  with patch.object(n.os,'close',interrupt):
   with self.assertRaises(KeyboardInterrupt) as caught:n.Native()
  self.assertTrue(caught.exception.worker_lifecycle['reaped']);self.assertFalse(caught.exception.worker_lifecycle['image_fd_pending']);self.assert_reaped()
 def test_authentication_failure_keeps_primary_and_fd_diagnostics(self):
  fcntl=n.fcntl.fcntl;close=os.close;before=set(os.listdir('/proc/self/fd'))
  def close_memfd(fd):
   if 'memfd:axon-candidate' in os.readlink('/proc/self/fd/'+str(fd)):raise OSError('close secondary')
   return close(fd)
  with patch.object(n.fcntl,'fcntl',side_effect=KeyboardInterrupt('seal primary')),patch.object(n.os,'close',side_effect=close_memfd):
   with self.assertRaises(KeyboardInterrupt) as caught:n.Native()
   self.assertTrue(caught.exception.worker_lifecycle['image_fd_pending']);self.assertTrue(n.OWNED)
  for worker in list(n.OWNED.values()):worker.abort()
  self.assertEqual(set(os.listdir('/proc/self/fd')),before)
 def test_context_registration_before_and_after_set(self):
  active=candidate.ACTIVE
  for after in (False,True):
   class Context:
    def get(self):return active.get()
    def set(self,value):
     if value is not None:
      if after:active.set(value)
      raise KeyboardInterrupt('registration')
     return active.set(value)
    def reset(self,token):return active.reset(token)
   with patch.object(candidate,'ACTIVE',Context()):
    with self.assertRaises(KeyboardInterrupt):
     with candidate.engine('native'):self.fail('entry succeeded')
   self.assertIsNone(active.get());self.assert_reaped()
 def test_context_exit_interruption_preserves_prior_error(self):
  active=candidate.ACTIVE
  class Context:
   def get(self):return active.get()
   def set(self,value):return active.set(value)
   def reset(self,token):active.reset(token);raise SystemExit('secondary exit')
  original=ValueError('primary conversion')
  with patch.object(candidate,'ACTIVE',Context()):
   with self.assertRaises(ValueError) as caught:
    with candidate.engine('native'):raise original
  self.assertIs(caught.exception,original);self.assertIn('context_reset',[x['operation'] for x in original.worker_lifecycle['cleanup_errors']]);self.assertIsNone(active.get());self.assert_reaped()
 def test_cleanup_multiple_failures_alive_retry_and_cumulative(self):
  w=n.Native();os.kill(w.proc.pid,signal.SIGSTOP)
  with patch.object(w.proc,'kill',side_effect=PermissionError('kill')),patch.object(w.proc,'wait',side_effect=subprocess.TimeoutExpired('worker',2)),patch.object(w.proc.stdin,'close',side_effect=OSError('stdin')),patch.object(w.proc.stdout,'close',side_effect=OSError('stdout')):
   first=w.abort();count=len(w.cleanup_errors);second=w.abort()
   self.assertIsNone(w.proc.poll());self.assertFalse(w.closed);self.assertFalse(second['reaped']);self.assertEqual(second['state'],'cleanup_incomplete');self.assertGreater(len(w.cleanup_errors),count);self.assertIn(w.worker_id,n.OWNED)
   with self.assertRaises(n.WorkerError):w.finish()
  errors=list(w.cleanup_errors);final=n.retry_cleanup(w.worker_id)
  self.assertTrue(final['reaped']);self.assertTrue(w.closed);self.assertEqual(w.cleanup_errors,errors)
  self.assertEqual(w.abort(),final);self.assertEqual(w.cleanup_errors,errors);self.assert_reaped()
 def test_wait_interrupted_after_real_reap_is_confirmed(self):
  w=n.Native();original=w.proc.wait
  def interrupted(*args,**kwargs):original(*args,**kwargs);raise KeyboardInterrupt('wait after reap')
  with patch.object(w.proc,'wait',interrupted):report=w.abort()
  self.assertTrue(report['reaped']);self.assertTrue(report['cleanup_complete']);self.assertIn('wait',[x['operation'] for x in report['cleanup_errors']]);self.assert_reaped()
 def test_pipe_close_failure_not_equal_to_reap(self):
  w=n.Native()
  with patch.object(w.proc.stdout,'close',side_effect=SystemExit('pipe close')):
   report=w.abort();self.assertTrue(report['reaped']);self.assertFalse(report['cleanup_complete']);self.assertFalse(w.closed)
  w.abort();self.assertTrue(w.closed);self.assertTrue(w.cleanup_errors);self.assert_reaped()
 def test_termination_interruption_recorded(self):
  w=n.Native()
  with patch.object(w.proc,'kill',side_effect=KeyboardInterrupt('terminate')),patch.object(w.proc,'wait',side_effect=subprocess.TimeoutExpired('worker',2)):
   report=w.abort();self.assertFalse(report['reaped']);self.assertTrue(report['cleanup_errors'])
  w.abort();self.assert_reaped()
 def test_unknown_process_status_retains_owner(self):
  w=n.Native()
  with patch.object(w.proc,'poll',side_effect=OSError('status unknown')),patch.object(w.proc,'wait',side_effect=OSError('wait unknown')):
   report=w.abort();self.assertEqual(report['process_state'],'termination_attempted_unconfirmed');self.assertFalse(w.closed)
  w.abort();self.assert_reaped()
 def test_preflight_report_includes_unresolved_cleanup(self):
  close=os.close
  def fail_image(fd):
   if 'memfd:axon-candidate' in os.readlink('/proc/self/fd/'+str(fd)):raise OSError('pending image')
   close(fd)
  with patch.object(n.os,'close',side_effect=fail_image):
   with self.assertRaises(LifecycleError) as caught:candidate.convert('unused','unused','unused','unused',selection='native')
   self.assertFalse(caught.exception.state['promoted']);self.assertIn('worker_lifecycle',caught.exception.state);self.assertTrue(caught.exception.state['worker_lifecycle']['image_fd_pending'])
  for worker in list(n.OWNED.values()):worker.abort()
  self.assert_reaped()

 def test_pipe_creation_interruption_closes_all_registered_fds(self):
  before=set(os.listdir('/proc/self/fd'));pipe=os.pipe2;count=0
  def interrupted(flags):
   nonlocal count
   count+=1
   if count==2:raise SystemExit('second pipe')
   return pipe(flags)
  with patch.object(n.os,'pipe2',interrupted):
   with self.assertRaises(SystemExit) as caught:n.Native()
  self.assertTrue(caught.exception.worker_lifecycle['cleanup_complete']);self.assertEqual(set(os.listdir('/proc/self/fd')),before)
 def test_auth_read_primary_not_masked_by_close_failure(self):
  before=set(os.listdir('/proc/self/fd'));read=os.read;close=os.close
  def badread(fd,size):
   if os.readlink('/proc/self/fd/'+str(fd)).endswith('/trust.json'):raise KeyboardInterrupt('auth read primary')
   return read(fd,size)
  def badclose(fd):
   if os.readlink('/proc/self/fd/'+str(fd)).endswith('/trust.json'):raise SystemExit('auth close secondary')
   close(fd)
  with patch.object(n.os,'read',badread),patch.object(n.os,'close',badclose):
   with self.assertRaises(KeyboardInterrupt) as caught:n.Native()
   self.assertEqual(str(caught.exception),'auth read primary');self.assertTrue(caught.exception.worker_lifecycle['pending_descriptors']);self.assertTrue(caught.exception.worker_lifecycle['cleanup_errors'])
  for worker in list(n.OWNED.values()):worker.abort()
  self.assertEqual(set(os.listdir('/proc/self/fd')),before)

 def test_real_sigint_during_cleanup_is_secondary_and_reaped(self):
  w=n.Native();kill=w.proc.kill
  def interrupted():kill();os.kill(os.getpid(),signal.SIGINT)
  with patch.object(w.proc,'kill',interrupted):report=w.abort()
  self.assertTrue(report['reaped']);self.assertTrue(report['cleanup_complete']);self.assertIn('cleanup_interrupt',[x['operation'] for x in report['cleanup_errors']]);self.assert_reaped()

 def test_parent_pipe_wrapper_failure_after_launch(self):
  fdopen=os.fdopen;before=set(os.listdir('/proc/self/fd'))
  for stop in (1,2):
   calls=0
   def interrupted(*args,**kwargs):
    nonlocal calls
    calls+=1
    if calls==stop:raise SystemExit('parent pipe wrapper')
    return fdopen(*args,**kwargs)
   with patch.object(n.os,'fdopen',interrupted):
    with self.assertRaises(SystemExit) as caught:n.Native()
   self.assertTrue(caught.exception.worker_lifecycle['reaped']);self.assertTrue(caught.exception.worker_lifecycle['cleanup_complete']);self.assertEqual(set(os.listdir('/proc/self/fd')),before)
  self.assert_reaped()

 def test_completed_cleanup_handle_cannot_target_new_worker(self):
  first=n.Native();old=first.worker_id;first.abort();second=n.Native()
  self.assertNotEqual(old,second.worker_id)
  with self.assertRaises(KeyError):n.retry_cleanup(old)
  self.assertIsNone(second.proc.poll());second.abort();self.assert_reaped()

class Reports(unittest.TestCase):
 setUp=Ownership.setUp
 tearDown=Ownership.tearDown
 assert_reaped=Ownership.assert_reaped
 def fixture(self):
  f=fixtures.ConversionTests();f.setUp();self.addCleanup(f.doCleanups);return f
 def test_conversion_primary_survives_unresolved_worker_cleanup(self):
  f=self.fixture();w=n.Native();token=n.ACTIVE.set(w)
  try:
   with patch.object(w,'_read',side_effect=n.WorkerError('primary IPC')),patch.object(w.proc,'kill',side_effect=OSError('kill')),patch.object(w.proc,'wait',side_effect=subprocess.TimeoutExpired('worker',2)),patch.object(w.proc.stdin,'close',side_effect=OSError('pipe')),patch.object(w.proc.stdout,'close',side_effect=OSError('pipe')):
    with self.assertRaises(LifecycleError) as caught:c.convert(str(f.source),str(f.parent),'result',f.anchor,0)
    report=caught.exception.state;self.assertEqual(report['error'],'primary IPC');self.assertFalse(report['promoted']);self.assertEqual(report['worker_lifecycle']['state'],'cleanup_incomplete');self.assertGreaterEqual(len(report['worker_lifecycle']['cleanup_errors']),4);self.assertFalse((f.parent/'result').exists())
  finally:n.ACTIVE.reset(token);w.abort()
 def test_context_cleanup_after_confirmed_publication(self):
  f=self.fixture();active=candidate.ACTIVE
  class Context:
   def get(self):return active.get()
   def set(self,value):return active.set(value)
   def reset(self,token):active.reset(token);raise KeyboardInterrupt('after publication')
  with patch.object(candidate,'ACTIVE',Context()):
   with self.assertRaises(LifecycleError) as caught:candidate.convert(str(f.source),str(f.parent),'result',f.anchor,selection='native')
  report=caught.exception.state;self.assertTrue(report['promoted']);self.assertEqual(report['durability'],'confirmed');self.assertTrue((f.parent/'result').exists());self.assertTrue(report['worker_lifecycle']['reaped']);self.assertTrue(report['worker_lifecycle']['cleanup_errors']);self.assert_reaped()

 def test_cli_reporting_failure_retains_primary_and_cleanup_state(self):
  f=self.fixture();w=n.Native();token=n.ACTIVE.set(w);out=io.StringIO();err=io.StringIO();convert=c.convert
  class Receipt:
   def __init__(self,*args):pass
   def __enter__(self):return self
   def __exit__(self,*args):pass
  def work(*args,**kwargs):return convert(*args,anchor=f.anchor,reserve_bytes=0,**kwargs)
  try:
   with patch.object(c,'SOURCE',str(f.source)),patch.object(c,'CONVERTED',str(f.parent)),patch.object(c,'convert',work),patch.object(c,'ReceiptDestination',Receipt),patch.object(c,'emit_receipt',side_effect=OSError('reporting')),patch.object(w,'_read',side_effect=n.WorkerError('primary IPC')),patch.object(w.proc,'kill',side_effect=OSError('kill')),patch.object(w.proc,'wait',side_effect=subprocess.TimeoutExpired('worker',2)),patch.object(w.proc.stdin,'close',side_effect=OSError('pipe')),patch.object(w.proc.stdout,'close',side_effect=OSError('pipe')),patch.object(sys,'stdout',out),patch.object(sys,'stderr',err),patch.object(sys,'argv',['tool','--artifact-name','result','--authorize-cpu-conversion','--receipt','synthetic']):
    self.assertEqual(c.main(),1)
   report=json.loads(err.getvalue());self.assertEqual(report['error'],'primary IPC');self.assertEqual(report['reporting_error_type'],'OSError');self.assertFalse(report['promoted']);self.assertEqual(report['worker_lifecycle']['state'],'cleanup_incomplete');self.assertGreaterEqual(len(report['worker_lifecycle']['cleanup_errors']),4)
  finally:n.ACTIVE.reset(token);w.abort()
  self.assert_reaped()
