"""CHILDLESS top-level authentication program. Never import as a library.
On *any* failure exit the process: the kernel closes even unregistered descriptors.
On success exec the authenticated supervisor. There is no Popen/fork here.
"""
import os,sys,json,hashlib,stat,fcntl
from pathlib import Path
ROOT=Path(__file__).absolute().parent
SEALS=fcntl.F_SEAL_WRITE|fcntl.F_SEAL_GROW|fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_SEAL
MODES={'tests','benchmark','scenario'}
SCENARIOS={'none','normal','before_ack','after_ack','partial','lost_ready','stop_before_ack','child_before_delivery','descriptor_checkpoint','kill_failure','wait_failure','supervisor_failure','bad_version','oversized','duplicate','before_go','unexpected_response','lost_ack','kill_unresolved','wait_unresolved','close_failure','partial_after_start','worker_crash','worker_signal'}
def read(fd,name,limit):
 parts=name.split('/')
 if any(p in ('','..','.') for p in parts):raise ValueError('unsafe source name')
 d=fd
 for part in parts[:-1]:d=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=d)
 f=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=d)
 st=os.fstat(f)
 if not stat.S_ISREG(st.st_mode) or st.st_nlink!=1 or st.st_uid!=os.getuid() or st.st_mode&0o022 or st.st_size>limit:raise ValueError('untrusted file')
 data=bytearray()
 while len(data)<=limit:
  b=os.read(f,min(65536,limit+1-len(data)))
  if not b:break
  data.extend(b)
 os.close(f)
 if d!=fd:os.close(d)
 if len(data)>limit:raise ValueError('oversized trusted input')
 return bytes(data)
def image(raw,label):
 f=os.memfd_create(label,os.MFD_ALLOW_SEALING|os.MFD_CLOEXEC)
 v=memoryview(raw)
 while v:
  n=os.write(f,v)
  if n<=0:raise OSError('short memfd write')
  v=v[n:]
 fcntl.fcntl(f,fcntl.F_ADD_SEALS,SEALS)
 if fcntl.fcntl(f,fcntl.F_GET_SEALS)!=SEALS:raise ValueError('missing seals')
 os.set_inheritable(f,True)
 return f
def main():
 mode=sys.argv[1] if len(sys.argv)>1 else 'tests';scenario=sys.argv[2] if len(sys.argv)>2 else 'none'
 if len(sys.argv)>3 or mode not in MODES or scenario not in SCENARIOS:raise ValueError('synthetic mode only')
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 if os.nice(0)<19:os.nice(19-os.nice(0))
 if os.geteuid()==0 or os.geteuid()!=os.getuid():raise ValueError('unprivileged operator required')
 if any(part in ('.','..') for part in ROOT.parts):raise ValueError('unsafe package path')
 d=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 for part in ROOT.parts[1:]:
  old=d;d=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=d);os.close(old)
  info=os.fstat(d)
  if info.st_uid not in (0,os.getuid()) or info.st_mode&0o022:raise ValueError('untrusted directory')
 raw=read(d,'trust.json',65536);trust=json.loads(raw)
 if trust['abi']!=2:raise ValueError('ABI mismatch')
 for name,digest in trust['sources'].items():
  if hashlib.sha256(read(d,name,2**20)).hexdigest()!=digest:raise ValueError('source authentication: '+name)
 images={}
 for name in ('worker','supervisor'):
  data=read(d,name,4*2**20)
  if hashlib.sha256(data).hexdigest()!=trust['executables'][name]:raise ValueError('executable authentication')
  images[name]=image(data,'axon-'+name)
 os.close(d)
 # Discard all other inherited descriptors. Fixed trusted exec argv/environment.
 keep=set(images.values())
 lower=3
 for fd in sorted(keep):
  os.closerange(lower,fd);lower=fd+1
 os.closerange(lower,1048576)
 fault=scenario if mode=='scenario' else 'none'
 os.execve('/proc/self/fd/'+str(images['supervisor']),['axon-supervisor',str(images['worker']),str(ROOT),mode,scenario,fault],{'LC_ALL':'C'})
if __name__=='__main__':
 try:main()
 except BaseException as error:
  try:os.write(2,(json.dumps({'status':'FAIL','phase':'childless_bootstrap','error_type':type(error).__name__,'error':str(error),'children_created':False})+'\n').encode())
  except BaseException:pass
  os._exit(1)
