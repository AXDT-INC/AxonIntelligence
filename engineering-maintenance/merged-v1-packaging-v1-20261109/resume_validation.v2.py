import json,sys,fcntl,subprocess,signal,datetime,traceback
from pathlib import Path
from campaign import stage
from runtime_safeguards import sha,write
P=Path(__file__).resolve().parent;ROOT=Path('/data/axonai')
def main():
 assert not (P/'RESUME2_COMPLETE.json').exists()
 assert (P/'baseline/RESULT.json').exists() and json.loads((P/'baseline/RESULT.json').read_text())['status']=='PASS'
 assert json.loads((P/'merged_repair1_CLEANUP.json').read_text())['owned_group_absent']
 assert __import__('os').environ.get('TMUX') and subprocess.check_output(['tmux','display-message','-p','#{session_attached}'],text=True).strip()=='0'
 lock=(ROOT/'engineering-maintenance/.main-sft-gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 now=datetime.datetime.now(datetime.timezone.utc);start=datetime.datetime.fromisoformat(json.loads((P/'PREFLIGHT.json').read_text())['utc']);assert (now-start).total_seconds()<3600 and now<datetime.datetime(2026,11,23,17,0,tzinfo=datetime.timezone.utc)
 for name,h in json.loads((P/'REPAIR2_EXECUTION_FREEZE.json').read_text()).items():assert sha(P/name)==h
 bindings=json.loads((P/'PREFLIGHT.json').read_text())['verified_source_files']
 for b in bindings:assert sha(b['path'])==b['sha256']
 write(P/'POSTREPAIR2_SOURCE_INTEGRITY.json',{'all_unchanged':True,'bindings':bindings})
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Repair validation 30-minute cap')));signal.alarm(1800)
 try:
  stage('merged_repair2',[sys.executable,'-B','-u',str(P/'validate_gpu.v3.py'),'merged','--output-name','merged-repair2'],900,gpu=True)
  stage('smoke',[sys.executable,'-B','-u',str(P/'validate_gpu.v3.py'),'smoke'],900,gpu=True)
  write(P/'RESUME2_COMPLETE.json',{'status':'PASS','stages':['merged_repair2','smoke'],'original_merge_repeated':False,'baseline_inference_repeated':False,'original_prompts_and_decoding_unchanged':True})
 finally:signal.alarm(0);lock.close()
if __name__=='__main__':
 try:main()
 except BaseException:(P/'RESUME2_TRACEBACK.log').write_text(traceback.format_exc());raise
