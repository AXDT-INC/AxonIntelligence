"""Coordinator proxy only. Child/PID/pipe acquisition belongs to native supervisor."""
import ctypes
import os,json,struct,select,time,math,contextvars,hashlib
from pathlib import Path
from audit_io import AuditError,Directory
ROOT=Path(__file__).absolute().parent
HEADER=struct.Struct('<6I');STATS=struct.Struct('<7Q3d');MAGIC=0x41584e31
KEYS=('elements','changed','zeroed_nonzero','output_subnormal','input_nonfinite','overflow','output_nonfinite','max_absolute_error','max_relative_error','sum_absolute_error')
ACTIVE=contextvars.ContextVar('candidate_engine',default=None)
SESSION=None
class WorkerError(AuditError):pass
class Session:
 def __init__(self):
  self.seq=0;self.token=0;self.failed=False;self.errors=[];self.cancelled_tokens=set()
  self.descriptors={fd:(os.fstat(fd).st_dev,os.fstat(fd).st_ino) for fd in (3,4)}
 def close(self):
  self.failed=True
  for fd,identity in list(self.descriptors.items()):
   try:
    st=os.fstat(fd)
    if (st.st_dev,st.st_ino)!=identity:
     self.descriptors.pop(fd);raise WorkerError('channel descriptor replaced; refusing to close alias')
    os.close(fd);self.descriptors.pop(fd)
   except BaseException as e:
    if isinstance(e,OSError) and e.errno==9:self.descriptors.pop(fd,None)
    else:
     try:message=str(e)
     except BaseException:message='<unprintable>'
     self.errors.append({'operation':'control_close','error_type':type(e).__name__,'error':message})
 def exchange(self,op,token,payload=b'',timeout=7):
  if op==4:self.cancelled_tokens.add(token)
  if self.failed:raise WorkerError('supervisor unavailable; outcome UNKNOWN; no retry')
  if len(payload)>131200:raise WorkerError('oversized frame')
  self.seq+=1;deadline=time.monotonic()+timeout
  def io(data=None,length=0):
   out=bytearray();view=memoryview(data) if data is not None else None
   while (len(view) if view is not None else length-len(out)):
    left=deadline-time.monotonic()
    if left<=0:raise WorkerError('supervisor timeout; outcome UNKNOWN')
    r,w,_=select.select([3] if view is None else [],[4] if view is not None else [],[],left)
    if not r and not w:raise WorkerError('supervisor timeout; outcome UNKNOWN')
    try:
     if view is not None:
      n=os.write(4,view)
      if n<=0:raise WorkerError('short control write')
      view=view[n:]
     else:
      b=os.read(3,length-len(out))
      if not b:raise WorkerError('supervisor EOF; outcome UNKNOWN')
      out.extend(b)
    except BlockingIOError:continue
   return bytes(out)
  try:
   io(HEADER.pack(0x41585332,2,self.seq,op,token,len(payload))+payload)
   h=HEADER.unpack(io(length=24))
   if h[:5]!=(0x41585332,2,self.seq,op|0x80000000,token) or h[5]>131200:raise WorkerError('malformed supervisor response')
   return io(length=h[5])
  except BaseException:
   # An ambiguous frame is never followed by another frame. EOF is cancellation.
   self.close();raise
class Native:
 def __init__(self,timeout=7):
  if SESSION is None or SESSION.failed:raise WorkerError('use childless bootstrap top-level entry point')
  self.timeout=timeout;self.seq=0;self.broken=False;self.cleanup_errors=[];self.context_pending=False
  # Resource cleanup is independent of successful protocol completion.
  self.session=SESSION;self.worker_pid=None
  self.finish_requested=False;self.finish_acknowledged=False;self.finish_validated=False;self.finish_response=None
  self._cancellation_occurred=False;self.numerical_work_completed=False
  self.publication_authorized=False;self.publication_state=None
  self.rename_target=None
  self.coordinator_phase='commit_not_requested';self.promotion_blocked=False;self.terminal_result=None
  self.commit_requested=False;self.commit_granted=False;self.commit_result_attempted=False;self.commit_result_confirmed=False
  SESSION.token+=1;self.worker_id=SESSION.token;self.info={'process_state':'UNKNOWN','reaped':False,'cleanup_complete':False,'pid':None,'token':self.worker_id};self.buffer=b''
  self.trust=json.loads((ROOT/'trust.json').read_text());self.start=time.perf_counter()
  self.publication_library=__import__('convert_fp16').load_publication_gate(self)
  self.gate_session=os.getpid();self.gate_generation=self.worker_id
  self.publication_gate=self.publication_library.gate_create(self.gate_session,self.gate_generation)
  if not self.publication_gate:raise WorkerError('native publication state allocation failed')
  try:
   self.control(1)
   if self.info['process_state']!='ready_unacknowledged':raise WorkerError('not READY')
   # Python ownership is committed before ACK; native ownership existed before READY.
   self.control(2)
   if self.info['process_state']!='running_supervised':raise WorkerError('not GO')
   self.startup_seconds=time.perf_counter()-self.start
  except BaseException as e:self.abort();self.attach(e);raise
 @property
 def cancellation_occurred(self):
  return self._cancellation_occurred or self.session.failed or self.worker_id in self.session.cancelled_tokens or self.info.get('cancellation_requested') is True
 def invalidate_publication(self):
  # Sticky for this worker. Cleanup success cannot undo cancellation/failure.
  self._cancellation_occurred=True;self.broken=True;self.publication_authorized=False
 def control(self,op,payload=b''):
  if op==4:self.invalidate_publication()
  try:
   if SESSION is not self.session or self.session.failed:raise WorkerError('worker session unavailable')
   if op==6:
    if self.broken or self.cancellation_occurred:raise WorkerError('cancelled worker cannot FINISH')
    if self.finish_requested:raise WorkerError('FINISH cannot be retried')
    self.finish_requested=True
   result=json.loads(self.session.exchange(op,self.worker_id,payload,timeout=self.timeout))
   if type(result) is not dict or type(result.get('token')) is not int or result['token']!=self.worker_id:raise WorkerError('worker token mismatch')
   pid=result.get('pid')
   if type(pid) is not int or pid<=0 or (self.worker_pid is not None and pid!=self.worker_pid):raise WorkerError('worker PID mismatch')
   if op==1:self.worker_pid=pid
   if result.get('commit_protocol')!=3:raise WorkerError('publication commit protocol mismatch')
   if type(result.get('cancellation_requested')) is not bool or type(result.get('cancellation_complete')) is not bool:raise WorkerError('invalid cancellation evidence')
   self.info=result
   # Session.exchange has checked magic, ABI, sequence, response op and token.
   if op==6:
    # Preserve the authenticated FINISH evidence independently of later STATUS
    # or CANCEL replies. A later clean status must not repair a failed FINISH.
    self.finish_response=json.loads(json.dumps(result))
    self.finish_acknowledged=True
    if not self.clean_completion():raise WorkerError('worker FINISH response is not a clean normal exit')
   return result
  except BaseException:
   self.invalidate_publication()
   if not self.info.get('reaped'):self.info.update(process_state='UNKNOWN',cleanup_complete=False)
   raise
 @property
 def closed(self):return bool(self.info.get('reaped') is True and self.info.get('cleanup_complete') is True and not self.context_pending)
 def exit_facts(self):
  status=self.info.get('wait_status')
  known=self.info.get('reaped') is True and type(status) is int and 0<=status<=65535
  normal=known and os.WIFEXITED(status)
  signaled=known and os.WIFSIGNALED(status)
  return {'worker_exited_normally':bool(normal),
          'worker_exit_code':os.WEXITSTATUS(status) if normal else None,
          'worker_exit_signal':os.WTERMSIG(status) if signaled else None,
          'worker_exited_unsuccessfully':bool(signaled or (normal and os.WEXITSTATUS(status)!=0)),
          'worker_exit_unknown':not (normal or signaled)}
 def clean_completion(self):
  facts=self.exit_facts()
  fields=('pid','token','process_state','reaped','cleanup_complete','wait_status',
          'cleanup_errors','error_count','errors_truncated_count')
  finish_matches=(self.finish_response is not None
                  and all(self.finish_response.get(k)==self.info.get(k) for k in fields))
  return (SESSION is self.session and not self.session.failed and not self.session.errors
          and not self.broken and not self.cancellation_occurred and self.closed and finish_matches
          and self.info.get('process_state')=='exited_reaped'
          and self.info.get('pid')==self.worker_pid and self.info.get('token')==self.worker_id
          and facts['worker_exited_normally'] and facts['worker_exit_code']==0
          and self.info.get('wait_status')==0
          and self.info.get('cleanup_errors')==[] and type(self.info.get('error_count')) is int and self.info['error_count']==0
          and type(self.info.get('errors_truncated_count')) is int and self.info['errors_truncated_count']==0 and not self.cleanup_errors
          and not self.buffer)
 def snapshot(self):
  state=self.publication_state
  info=dict(self.info)
  # Caller-visible cancellation mirror may be newer, but never clears native history.
  return dict(info,**self.exit_facts(),worker_id=self.worker_id,unusable=self.broken,
              supervisor_available=not self.session.failed,local_cleanup_errors=list(self.cleanup_errors),
              session_cleanup_errors=list(self.session.errors),
              pending_channel_descriptors=list(self.session.descriptors) if self.session.failed else [],
              commit_requested=self.commit_requested,commit_granted=self.commit_granted,
              commit_result_attempted=self.commit_result_attempted,commit_result_confirmed=self.commit_result_confirmed,
              numerical_work_completed=self.numerical_work_completed,finish_requested=self.finish_requested,
              finish_acknowledged=self.finish_acknowledged,finish_validated=self.finish_validated,
              finish_response=json.loads(json.dumps(self.finish_response)),
              cancellation_occurred=self.cancellation_occurred,
              publication_authorized=bool(self.publication_authorized and self.finish_validated and self.numerical_work_completed
                                         and state is not None and state.get('promoted') is False
                                         and state.get('phase') in ('numerical_verification','pre_publication')
                                         and 'primary_failure' not in state and state.get('status')!='FAIL' and self.clean_completion()),
              publication_already_confirmed=state is not None and state.get('promoted') is True,
              publication_outcome=state.get('promoted') if state is not None else False,
              publication_durability=state.get('durability') if state is not None else 'not_published')
 def diagnostic(self,operation,error):
  try:message=str(error)
  except BaseException:message='<unprintable>'
  self.cleanup_errors.append({'operation':operation,'error_type':type(error).__name__,'error':message})
 def attach(self,error):
  try:
   error.worker_lifecycle=self.snapshot()
   if hasattr(error,'state'):error.state['worker_lifecycle']=self.snapshot()
  except BaseException as secondary:self.diagnostic('attach',secondary)
 def _write(self,b):
  if self.finish_requested:raise WorkerError('no numerical work after FINISH request')
  self.numerical_work_completed=False;self.publication_authorized=False
  if self.buffer:raise WorkerError('unexpected additional response')
  self.buffer=SESSION.exchange(5,self.worker_id,b,timeout=self.timeout)
 def _read(self,n):
  if len(self.buffer)<n:raise WorkerError('truncated numerical response')
  out=self.buffer[:n];self.buffer=self.buffer[n:];return out
 def abort(self):
  self.invalidate_publication()
  if not self.session.failed and (not self.closed or self.info.get('cancellation_requested') is not True):
   try:self.control(4)
   except BaseException as e:self.diagnostic('cancel',e)
  return self.snapshot()
 def finish(self):
  try:
   # An idempotent return requires a previously validated FINISH, never merely
   # reaping or descriptor cleanup. Revalidate eligibility after later failures.
   if self.finish_validated:
    if not self.clean_completion():raise WorkerError('successful FINISH no longer eligible for publication')
    return
   if self.broken or self.cancellation_occurred:raise WorkerError('failed worker cannot authorize promotion')
   if self.buffer:raise WorkerError('trailing numerical response')
   if not self.finish_requested:self.control(6)
   if not self.finish_acknowledged or not self.clean_completion():raise WorkerError('worker FINISH not confirmed clean')
   self.finish_validated=True
  except BaseException as e:self.abort();self.attach(e);raise
 def authorize_publication(self,result,lifecycle):
  # Only the coordinator supplies the full independent verification result.
  # FINISH alone (including a standalone verification session) is insufficient.
  try:
   if type(lifecycle) is not dict or lifecycle.get('promoted') is not False:raise WorkerError('publication outcome is not unpublished')
   if self.publication_state is not None and self.publication_state is not lifecycle:raise WorkerError('worker cannot authorize another artifact')
   self.publication_state=lifecycle
   if (type(result) is not dict or result.get('status')!='PASS'
       or result.get('source_integrity_rechecked') is not True
       or result.get('complete_output_inventory_verified') is not True
       or lifecycle.get('numerical_verification')!='PASS'):raise WorkerError('complete numerical verification required')
   self.numerical_work_completed=True
   self.finish()
   self.publication_authorized=True
   self.require_publication_authorized(lifecycle)
  except BaseException as e:self.abort();self.attach(e);raise
 def require_publication_authorized(self,lifecycle):
  if (self.publication_state is not lifecycle or lifecycle.get('promoted') is not False
      or lifecycle.get('phase') not in ('numerical_verification','pre_publication')
      or 'primary_failure' in lifecycle or lifecycle.get('status')=='FAIL'
      or not self.numerical_work_completed or lifecycle.get('numerical_verification')!='PASS'
      or not self.finish_requested or not self.finish_acknowledged or not self.finish_validated
      or not self.publication_authorized or not self.clean_completion()):
   self.invalidate_publication()
   raise WorkerError('publication authorization denied')
 def begin_commit(self,lifecycle):
  self.require_publication_authorized(lifecycle)
  if self.commit_requested:raise WorkerError('commit cannot be retried')
  self.commit_requested=True
  response=self.control(7)
  if (response.get('commit_state')!='in_progress' or response.get('commit_decision')!='granted'
      or response.get('cancellation_requested') is not False or response.get('finish_completed') is not True
      or not self.clean_completion()):
   self.invalidate_publication();raise WorkerError('supervisor denied publication commit')
  self.commit_granted=True
  if self.publication_library.gate_arm(self.publication_gate,self.gate_session,self.gate_generation):raise WorkerError('native publication authorization denied')
  self.coordinator_phase='promotion_pending'
  lifecycle['commit_protocol']=self.commit_snapshot()
 def invalidate_native_publication(self):
  if self.publication_library.gate_invalidate(self.publication_gate,self.gate_session,self.gate_generation):raise WorkerError('native invalidation identity failure')
 def native_publication_status(self):
  out=(ctypes.c_uint64*7)()
  if self.publication_library.gate_status(self.publication_gate,self.gate_session,self.gate_generation,out):raise WorkerError('native status identity failure')
  return dict(zip(('session','generation','eligible','invalidated','attempted','confirmed','errno'),out))
 def commit_snapshot(self):
  return dict(native_publication=self.native_publication_status(),coordinator_phase=self.coordinator_phase,promotion_blocked=self.promotion_blocked,terminal_result=self.terminal_result,requested=self.commit_requested,granted=self.commit_granted,
              result_attempted=self.commit_result_attempted,result_confirmed=self.commit_result_confirmed,
              supervisor_state=('UNKNOWN' if (self.session.failed or self.commit_result_attempted) and not self.commit_result_confirmed else self.info.get('commit_state','UNKNOWN')),
              last_observed_supervisor_state=self.info.get('commit_state','UNKNOWN'),
              supervisor_available=not self.session.failed,
              cancellation_requested=self.info.get('cancellation_requested'),
              cancellation_complete=self.info.get('cancellation_complete'),
              publication_outcome_known_to_supervisor=self.commit_result_confirmed and self.info.get('commit_state') in ('published','not_published'))
 def start_promotion(self,lifecycle):
  if (lifecycle is not self.publication_state or not self.commit_granted
      or self.coordinator_phase!='promotion_pending' or self.promotion_blocked
      or self.terminal_result is not None or self.commit_result_attempted):
   raise WorkerError('promotion is not pending or has been finalized/blocked')
  self.coordinator_phase='rename_attempted'
 def publication_observed(self,lifecycle):
  if lifecycle is not self.publication_state or self.coordinator_phase!='rename_attempted':
   raise WorkerError('unexpected publication transition')
  self.coordinator_phase='promotion_confirmed'
 def settle_publication(self,lifecycle,failed=False):
  # Called by the coordinator only after the promotion operation returned or
  # unwound. Once latched, this outcome cannot return to the promotion path.
  if not self.commit_requested:return
  if lifecycle is not self.publication_state:raise WorkerError('different commit lifecycle')
  if self.terminal_result is not None:return
  promoted=lifecycle.get('promoted');durability=lifecycle.get('durability')
  if not failed and (promoted is not True or durability!='confirmed' or self.coordinator_phase!='promotion_confirmed'):
   raise WorkerError('publication operation has not terminated')
  if promoted is False:result=(0,0);phase='not_published'
  elif promoted is True:result=(1,2 if durability=='confirmed' else 1);phase='durability_confirmed' if result[1]==2 else 'durability_failed_or_unknown'
  else:result=(2,3);phase='promotion_outcome_unknown'
  self.invalidate_native_publication()
  self.promotion_blocked=True;self.terminal_result=result;self.coordinator_phase=phase
 def complete_commit(self,lifecycle,primary=None):
  if not self.commit_requested:return
  try:
   if lifecycle is not self.publication_state:
    self.invalidate_native_publication()
    self.promotion_blocked=True
    raise WorkerError('different commit lifecycle')
   if self.terminal_result is None:
    self.invalidate_native_publication()
    self.promotion_blocked=True
    raise WorkerError('premature commit completion: publication operation still pending')
   if not self.commit_granted:raise WorkerError('commit grant unknown; no publication/result retry')
   promoted=lifecycle.get('promoted');durability=lifecycle.get('durability')
   if promoted is False:outcome,durable=0,0
   elif promoted is True:outcome,durable=1,2 if durability=='confirmed' else 1
   else:outcome,durable=2,3
   if (outcome,durable)!=self.terminal_result:raise WorkerError('conflicting terminal completion')
   if self.commit_result_attempted:
    if self.commit_result_confirmed:return
    raise WorkerError('commit result unknown; no retry')
   self.commit_result_attempted=True
   response=self.control(8,struct.pack('<2I',outcome,durable))
   expected=('not_published','published','UNKNOWN')[outcome]
   if response.get('commit_state')!=expected or response.get('commit_durability')!=durable:raise WorkerError('invalid commit result acknowledgment')
   self.commit_result_confirmed=True
   self.coordinator_phase='commit_result_finalized'
  except BaseException as secondary:
   self.diagnostic('commit_result',secondary)
   if primary is None:raise
   self.attach(primary)
  finally:
   lifecycle['commit_protocol']=self.commit_snapshot()
 def release_if_complete(self):pass
 def block(self,raw,stats,actual=None):
  if self.closed or self.broken:raise WorkerError('worker unavailable; no retry')
  if type(raw) is not bytes or len(raw)%2 or len(raw)>131072:raise WorkerError('invalid coordinator block')
  if actual is not None and (type(actual) not in (bytes,bytearray) or len(actual)!=len(raw)):raise WorkerError('verification length')
  local=dict(stats);output=[]
  try:
   for pos in range(0,len(raw),65536):
    b=raw[pos:pos+65536];a=None if actual is None else actual[pos:pos+65536]
    self.seq+=1;op=1 if a is None else 2;self.deadline=time.monotonic()+self.timeout
    self._write(HEADER.pack(MAGIC,1,self.seq,op,len(b),80)+STATS.pack(*(local[k] for k in KEYS))+b+(a or b''))
    h=HEADER.unpack(self._read(24))
    if h[:3]!=(MAGIC,1,self.seq) or h[5]!=80 or h[3]>6 or h[4]!=(len(b) if op==1 and h[3]==0 else 0):raise WorkerError('malformed response')
    result=dict(zip(KEYS,STATS.unpack(self._read(80))))
    if h[3]:raise WorkerError('native numerical rejection '+str(h[3]))
    if result['elements']!=local['elements']+len(b)//2 or any(result[k]>result['elements'] or result[k]<local[k] for k in KEYS[1:7]) or any(not math.isfinite(result[k]) or result[k]<local[k] for k in KEYS[7:]) or any(result[k] for k in KEYS[4:7]):raise WorkerError('malformed statistics')
    local=result
    if h[4]:output.append(self._read(h[4]))
   stats.update(local);return b''.join(output)
  except BaseException as error:self.abort();self.attach(error);raise
def record_worker_state(state,error=None):
 worker=ACTIVE.get();report=worker.snapshot() if worker else getattr(error,'worker_lifecycle',None)
 if report is not None:state['worker_lifecycle']=report
def finish_before_publication(result=None,lifecycle=None):
 if ACTIVE.get() is not None:ACTIVE.get().authorize_publication(result,lifecycle)
def require_publication_authorized(lifecycle):
 if ACTIVE.get() is not None:ACTIVE.get().begin_commit(lifecycle)
def complete_publication(lifecycle,primary=None):
 if ACTIVE.get() is not None:ACTIVE.get().complete_commit(lifecycle,primary)
def provenance():
 if ACTIVE.get() is None:return {'engine':'v3-reference','abi':None,'executable_sha256':None,'trust_sha256':None}
 raw=(ROOT/'trust.json').read_bytes();t=json.loads(raw)
 return {'engine':'native-supervisor-v2','abi':2,'executable_sha256':t['executables']['worker'],'supervisor_sha256':t['executables']['supervisor'],'trust_sha256':hashlib.sha256(raw).hexdigest()}
def validate_provenance(record):
 if not isinstance(record,dict):raise WorkerError('missing candidate provenance')
 if record.get('engine')=='v3-reference':expected={'engine':'v3-reference','abi':None,'executable_sha256':None,'trust_sha256':None}
 else:
  with Directory(str(ROOT)) as d:raw=d.read('trust.json',65536)
  t=json.loads(raw);expected={'engine':'native-supervisor-v2','abi':2,'executable_sha256':t['executables']['worker'],'supervisor_sha256':t['executables']['supervisor'],'trust_sha256':hashlib.sha256(raw).hexdigest()}
 if record!=expected:raise WorkerError('candidate provenance mismatch')

def perform_promotion(lifecycle,operation,*args):
 worker=ACTIVE.get()
 if worker is not None:worker.start_promotion(lifecycle)
 return operation(*args)
def publication_observed(lifecycle):
 if ACTIVE.get() is not None:ACTIVE.get().publication_observed(lifecycle)
def settle_publication(lifecycle,failed=False):
 if ACTIVE.get() is not None:ACTIVE.get().settle_publication(lifecycle,failed)
