import os,sys,json,time,datetime,fcntl,subprocess,signal,shutil,traceback
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=Path('/data/axonai');M=ROOT/'models/merged/AxonAI-v1-27B'
from runtime_safeguards import write,append,sha

def stop(p):
 try:os.killpg(p.pid,signal.SIGTERM)
 except ProcessLookupError:pass
 try:p.wait(timeout=10)
 except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)

def stage(name,cmd,timeout,gpu=False):
 start=time.monotonic();p=None
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1' if gpu else '',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(P/'tmp'),XDG_CACHE_HOME=str(P/'cache'),CUDA_CACHE_PATH=str(P/'cache/cuda'),TRITON_CACHE_DIR=str(P/'cache/triton'),OMP_NUM_THREADS='16' if not gpu else '8')
 try:
  with (P/(name+'.log')).open('xb',buffering=0) as log:
   p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);write(P/(name+'_PROCESS.json'),{'pid':p.pid,'command':cmd})
   while p.poll() is None:
    assert time.monotonic()-start<timeout,'Stage runtime cap'
    status=Path(f'/proc/{p.pid}/status')
    rss=0
    if status.exists():
     for line in status.read_text().splitlines():
      if line.startswith('VmRSS:'):rss=int(line.split()[1])*1024
    assert rss<256*1024**3,'CPU RSS budget exceeded'
    mem=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'));assert mem>250*1024**3,'Host headroom lost'
    row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rss_bytes':rss,'mem_available_bytes':mem}
    if gpu:row['gpu_csv']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True,timeout=10)
    append(P/(name+'_RESOURCES.jsonl'),row);time.sleep(2)
   assert p.returncode==0,f'{name} failed with {p.returncode}'
 finally:
  if p:stop(p)
  absent=True
  if p:
   try:os.killpg(p.pid,0);absent=False
   except ProcessLookupError:pass
  write(P/(name+'_CLEANUP.json'),{'worker_pid':p.pid if p else None,'owned_group_absent':absent,'exit_code':p.returncode if p else None,'elapsed_seconds':time.monotonic()-start})

def main():
 assert os.environ.get('TMUX') and subprocess.check_output(['tmux','display-message','-p','#{session_attached}'],text=True).strip()=='0'
 assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime(2026,11,23,17,0,tzinfo=datetime.timezone.utc)
 lock=(ROOT/'engineering-maintenance/.main-sft-gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 pre=json.loads((P/'PREFLIGHT.json').read_text());assert pre['all_passed'] and not M.exists()
 freeze=json.loads((P/'EXECUTION_FREEZE.json').read_text())
 for n,h in freeze.items():assert sha(P/n)==h,n
 for d in ['tmp','cache']:(P/d).mkdir(exist_ok=True)
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Campaign 90-minute cap')));signal.alarm(5400)
 try:
  stage('merge_cpu',[sys.executable,'-B','-u',str(P/'merge_cpu.py')],1800)
  assert json.loads((P/'MERGE_RESULT.json').read_text())['status']=='PASS'
  M.parent.mkdir(exist_ok=True);(P/'merged-build').rename(M)
  stage('verify_weights',[sys.executable,'-B','-u',str(P/'verify_weights.py')],600)
  for name in ['baseline','merged','smoke']:stage(name,[sys.executable,'-B','-u',str(P/'validate_gpu.py'),name],900,gpu=True)
  write(P/'CAMPAIGN_COMPLETE.json',{'status':'PASS','training':False,'stages':['merge_cpu','baseline','merged','smoke'],'gpu_count':2})
 finally:signal.alarm(0);lock.close()
if __name__=='__main__':
 try:main()
 except BaseException:(P/'CAMPAIGN_TRACEBACK.log').write_text(traceback.format_exc());raise
