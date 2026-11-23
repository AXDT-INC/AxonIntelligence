"""Linux-only bounded supervisor; no native code is loaded in the coordinator."""
import os,stat,json,hashlib,fcntl,subprocess,select,struct,time,math,contextvars,errno,uuid
from lifecycle_support import acquisition_signals, OwnedDescriptors
from pathlib import Path
from audit_io import AuditError,Directory,canonical
ROOT=Path(__file__).absolute().parent
HEADER=struct.Struct('<6I');STATS=struct.Struct('<7Q3d');MAGIC=0x41584e31
KEYS=('elements','changed','zeroed_nonzero','output_subnormal','input_nonfinite','overflow','output_nonfinite','max_absolute_error','max_relative_error','sum_absolute_error')
ACTIVE=contextvars.ContextVar('candidate_engine',default=None)
class WorkerError(AuditError):pass
# Strong ownership persists until resource cleanup is confirmed. This is also
# the explicit recovery handle when an exception interrupts construction.
OWNED = {}

def record_worker_state(state, error=None):
 worker = ACTIVE.get()
 report = worker.snapshot() if worker is not None else getattr(error,'worker_lifecycle',None)
 if report is not None:state['worker_lifecycle'] = report

def retry_cleanup(worker_id):
 worker = OWNED[worker_id]
 worker.abort()
 return worker.snapshot()


def authenticated_image(owner=None):
 # Same descriptor-relative/no-follow trust policy; explicit ownership now covers
 # every ancestor and input descriptor, including authentication failure paths.
 errors=[]
 def local_diagnostic(operation,error):
  errors.append({'operation':operation,'error_type':type(error).__name__})
 ledger=owner.descriptors if owner is not None else OwnedDescriptors(local_diagnostic)
 try:
  parts=str(ROOT).split('/')
  if not str(ROOT).startswith('/') or any(x in ('.','..') for x in parts):raise WorkerError('unsafe trusted directory')
  current=ledger.acquire(lambda:os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW))
  for part in filter(None,parts):
   parent=current
   current=ledger.acquire(lambda:os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent))
   ledger.close(parent,strict=True)
  def read(name,limit):
   fd=ledger.acquire(lambda:os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=current))
   try:
    info=os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:raise WorkerError('not a unique regular trusted file')
    if info.st_uid!=os.getuid() or info.st_mode&0o022:raise WorkerError('untrusted owner/mode')
    out=bytearray()
    while len(out)<=limit:
     block=os.read(fd,min(65536,limit+1-len(out)))
     if not block:break
     out.extend(block)
    if len(out)>limit:raise WorkerError('oversized trusted file')
   except BaseException:
    ledger.close(fd);raise
   ledger.close(fd,strict=True)
   return bytes(out)
  trust_bytes=read('trust.json',16384)
  trust=json.loads(trust_bytes)
  trust['_authenticated_manifest_sha256']=hashlib.sha256(trust_bytes).hexdigest()
  if trust['abi']!=1:raise WorkerError('ABI mismatch')
  for name in ('worker.c','convert.c','verify.c','prototype.h'):
   if hashlib.sha256(read(name,1024*1024)).hexdigest()!=trust['sources'][name]:raise WorkerError('source authentication')
  binary=read('worker',4*1024*1024)
  if hashlib.sha256(binary).hexdigest()!=trust['executable_sha256']:raise WorkerError('executable authentication')
  ledger.close(current,strict=True)
  fd=ledger.acquire(lambda:os.memfd_create('axon-candidate',os.MFD_ALLOW_SEALING|os.MFD_CLOEXEC))
  if owner is not None:owner.own_image(fd)
  view=memoryview(binary)
  while view:
   count=os.write(fd,view)
   if count<=0:raise WorkerError('memfd short write')
   view=view[count:]
  fcntl.fcntl(fd,fcntl.F_ADD_SEALS,fcntl.F_SEAL_WRITE|fcntl.F_SEAL_GROW|fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_SEAL)
  os.lseek(fd,0,0)
  if owner is None:ledger.fds.pop(fd) # explicit ownership transfer to standalone caller
  return fd,trust
 except BaseException as error:
  ledger.close_all()
  if owner is None:error.worker_lifecycle={'state':'cleanup_incomplete' if ledger.fds else 'not_launched','pending_descriptors':list(ledger.fds),'cleanup_errors':errors}
  raise

class Native:
 def __init__(self,timeout=5.0):
  if not 0<timeout<=30:raise ValueError('timeout bound')
  self.timeout=timeout;self.seq=0;self.broken=False;self.proc=None
  self.cleanup_errors=[];self.image_fd=None;self.image_identity=None
  self.descriptors=OwnedDescriptors(self.diagnostic)
  self.process_state='not_launched';self.context_pending=False
  self.state_history=['not_launched'];self.worker_id=uuid.uuid4().hex
  self.start=time.perf_counter()
  OWNED[self.worker_id]=self
  try:
   with acquisition_signals():
    fd,self.trust=authenticated_image(owner=self)
    if self.image_fd is None:self.own_image(fd)
    # Allocate and register the process object BEFORE launch. If Popen.__init__
    # raises after a child exists, its PID/pipes remain reachable by this owner.
    child_in,parent_in=self.descriptors.pipe()
    parent_out,child_out=self.descriptors.pipe()
    devnull=self.descriptors.acquire(lambda:os.open('/dev/null',os.O_WRONLY|os.O_CLOEXEC))
    self.proc=subprocess.Popen.__new__(subprocess.Popen)
    subprocess.Popen.__init__(self.proc,['/proc/self/fd/'+str(fd)],stdin=child_in,stdout=child_out,stderr=devnull,pass_fds=(fd,),close_fds=True,cwd=ROOT,env={'LC_ALL':'C'},bufsize=0)
    self.transition('running_supervised')
    # The parent ends remain in the descriptor ledger until their stream owner
    # is registered; no child or pipe depends on Popen returning successfully.
    self.proc.stdin=os.fdopen(parent_in,'wb',buffering=0)
    self.descriptors.fds.pop(parent_in)
    self.proc.stdout=os.fdopen(parent_out,'rb',buffering=0)
    self.descriptors.fds.pop(parent_out)
    for owned in (child_in,child_out,devnull):self.descriptors.close(owned,strict=True)
    if not self.close_image(strict=True):raise WorkerError('executable descriptor cleanup incomplete')
   os.set_blocking(self.proc.stdin.fileno(),False);os.set_blocking(self.proc.stdout.fileno(),False)
   self.deadline=time.monotonic()+self.timeout
   if self._read(24)!=HEADER.pack(MAGIC,1,0,0,0,80):raise WorkerError('invalid worker greeting')
   self.startup_seconds=time.perf_counter()-self.start
  except BaseException as error:
   self.abort();self.attach(error);raise
 def transition(self,state):
  self.process_state=state
  self.state_history.append(state)
 def diagnostic(self,operation,error):
  try:message=str(error)
  except BaseException:message='<unprintable>'
  self.cleanup_errors.append({'operation':operation,'error_type':type(error).__name__,'error':message})
 def own_image(self,fd):
  self.image_fd=fd
  if fd not in self.descriptors.fds:
   self.descriptors.fds[fd]=None
   info=os.fstat(fd);self.descriptors.fds[fd]=(info.st_dev,info.st_ino)
 def close_image(self,strict=False):
  if self.image_fd is None:return True
  try:return self.descriptors.close(self.image_fd,strict=strict)
  finally:
   if self.image_fd not in self.descriptors.fds:self.image_fd=None
 def resource_status(self):
  pipes={}
  for name in ('stdin','stdout','stderr'):
   pipe=getattr(self.proc,name,None)
   pipes[name]='closed' if pipe is None or pipe.closed else 'open'
  return pipes
 @property
 def closed(self):
  return self.process_state in ('not_launched','exited_reaped') and not self.descriptors.fds and all(v=='closed' for v in self.resource_status().values()) and not self.context_pending
 def snapshot(self):
  complete=self.closed
  return {'worker_id':self.worker_id,'pid':getattr(self.proc,'pid',None),'state':'cleanup_incomplete' if self.broken and not complete else self.process_state,'process_state':self.process_state,'unusable':self.broken,'cleanup_complete':complete,'reaped':self.process_state=='exited_reaped','returncode':getattr(self.proc,'returncode',None),'pipes':self.resource_status(),'image_fd_pending':self.image_fd in self.descriptors.fds,'pending_descriptors':list(self.descriptors.fds),'context_pending':self.context_pending,'cleanup_errors':[dict(e) for e in self.cleanup_errors],'state_history':list(self.state_history)}
 def attach(self,error):
  error.worker_lifecycle=self.snapshot()
  if hasattr(error,'state'):error.state['worker_lifecycle']=self.snapshot()
 def release_if_complete(self):
  if self.closed:OWNED.pop(self.worker_id,None)
 def _wait(self,fd,write=False):
  left=self.deadline-time.monotonic()
  if left<=0:raise WorkerError('worker timeout')
  ready=select.select([] if write else [fd],[fd] if write else [],[],left)
  if not (ready[0] or ready[1]):raise WorkerError('worker timeout')
 def _read(self,n):
  out=bytearray()
  while len(out)<n:
   self._wait(self.proc.stdout.fileno())
   try:b=os.read(self.proc.stdout.fileno(),n-len(out))
   except BlockingIOError:continue
   if not b:raise WorkerError('worker premature exit: '+str(self.proc.poll()))
   out.extend(b)
  return bytes(out)
 def _write(self,b):
  v=memoryview(b)
  while v:
   self._wait(self.proc.stdin.fileno(),True)
   try:n=os.write(self.proc.stdin.fileno(),v)
   except BlockingIOError:continue
   if n<=0:raise WorkerError('worker pipe short write')
   v=v[n:]
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
 def abort(self):
  try:
   with acquisition_signals():return self._abort_resources()
  except BaseException as cleanup:
   self.broken=True;self.diagnostic('cleanup_interrupt',cleanup)
   if not self.closed:OWNED[self.worker_id]=self
   return self.snapshot()
 def _abort_resources(self):
  if self.broken and self.closed:
   self.release_if_complete();return self.snapshot()
  OWNED[self.worker_id]=self
  self.broken=True
  self.state_history.append('unusable_failed')
  pid=getattr(self.proc,'pid',None)
  if pid is not None and self.process_state!='exited_reaped':
   self.transition('termination_requested')
   try:
    if self.proc.poll() is None:self.proc.kill()
   except BaseException as error:self.diagnostic('terminate',error)
   self.transition('termination_attempted_unconfirmed')
   try:
    self.proc.wait(timeout=2)
    self.transition('exited_reaped')
   except BaseException as error:
    self.diagnostic('wait',error)
    try:
     if self.proc.poll() is not None:self.transition('exited_reaped')
    except BaseException as probe:self.diagnostic('process_status',probe)
  elif pid is None:
   self.transition('not_launched')
  for name in ('stdin','stdout','stderr'):
   pipe=getattr(self.proc,name,None)
   if pipe is not None and not pipe.closed:
    try:pipe.close()
    except BaseException as error:self.diagnostic(name+'_close',error)
  self.close_image()
  self.descriptors.close_all()
  self.release_if_complete()
  return self.snapshot()
 def finish(self):
  if self.broken:
   error=WorkerError('failed worker cannot authorize promotion');self.attach(error);raise error
  if self.closed:return
  try:
   self.seq+=1;self.deadline=time.monotonic()+self.timeout
   self._write(HEADER.pack(MAGIC,1,self.seq,3,0,80))
   if self._read(24)!=HEADER.pack(MAGIC,1,self.seq,0,0,80):raise WorkerError('bad shutdown acknowledgement')
   self.proc.stdin.close()
   code=self.proc.wait(timeout=self.timeout)
   self.transition('exited_reaped')
   if code!=0:raise WorkerError('worker nonzero exit')
   if os.read(self.proc.stdout.fileno(),1):raise WorkerError('trailing worker response')
   self.proc.stdout.close()
   if not self.close_image(strict=True):raise WorkerError('descriptor cleanup incomplete')
   self.release_if_complete()
  except BaseException as error:self.abort();self.attach(error);raise

def finish_before_publication():
 e=ACTIVE.get()
 if e is not None:e.finish()
def provenance():
 e=ACTIVE.get()
 return {'engine':'v3-reference' if e is None else 'native-supervised-v1','abi':None if e is None else 1,'executable_sha256':None if e is None else e.trust['executable_sha256'],'trust_sha256':e.trust['_authenticated_manifest_sha256'] if e else None}

def validate_provenance(record):
 if not isinstance(record,dict):raise WorkerError('missing candidate provenance')
 if record.get('engine')=='v3-reference':
  expected={'engine':'v3-reference','abi':None,'executable_sha256':None,'trust_sha256':None}
 elif record.get('engine')=='native-supervised-v1':
  with Directory(str(ROOT)) as d:raw=d.read('trust.json',16384)
  trust=json.loads(raw)
  expected={'engine':'native-supervised-v1','abi':1,'executable_sha256':trust['executable_sha256'],'trust_sha256':hashlib.sha256(raw).hexdigest()}
 else:raise WorkerError('unknown recorded engine')
 if record!=expected:raise WorkerError('candidate provenance mismatch')
