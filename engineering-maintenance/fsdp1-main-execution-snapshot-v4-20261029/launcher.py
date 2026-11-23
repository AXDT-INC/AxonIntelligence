"""Main-only launcher for the new hash-bound snapshot. Must execute inside detached tmux."""
import argparse,fcntl,json,os,subprocess,sys,math
from pathlib import Path
from guard import ROOT,sha,verify_approval,require
from supervisor import supervise
from shared_engine import detect_conflict
PYTHON=Path('/data/axonai/.venv/bin/python')

def atomic(path,value):
    temp=path.with_suffix('.pending')
    with temp.open('x') as f:
        json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def command(output,contract):
    b=contract['bindings']
    return [str(PYTHON),'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(ROOT/'train.py'),'--run-dir',str(output),'--config',str(ROOT/'config.fsdp1.json'),'--train-file',b['dataset']['path'],'--evaluation-waiver',b['waiver']['path']]

def prepare(a,c,approval,digest):
    out=Path(a['output']);out.mkdir(exist_ok=False)
    b=c['bindings']
    identity={'mode':'main','config_sha256':a['config_sha256'],'model_revision':a['model_revision'],
        'dataset_manifest_sha256':a['dataset_sha256'],'extracted_jsonl_sha256':a['training_jsonl_sha256'],
        'evaluation_waiver_sha256':b['waiver']['sha256'],'approval_sha256':digest,
        'snapshot_sha256':a['snapshot_manifest']['sha256'],'code':json.loads((ROOT/'SNAPSHOT.json').read_text())['files']}
    atomic(out/'identity.json',identity)
    atomic(out/'freeze.json',{'splits':{'train':b['dataset']}})
    atomic(out/'LAUNCH_AUTHORIZATION.json',{'mode':'main','identity_sha256':sha(out/'identity.json'),
        'approval':str(Path(approval).resolve()),'approval_sha256':digest,'extraction_receipt':b['extraction']['path']})
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--approval',required=True);p.add_argument('--approval-sha256',required=True);p.add_argument('--preflight',action='store_true');args=p.parse_args()
    a,c=verify_approval(args.approval,args.approval_sha256)
    from train import evaluation_policy
    evaluation_policy('main',None,c['bindings']['waiver']['path'],{'dataset_manifest_sha256':a['dataset_sha256'],'evaluation_waiver_sha256':c['bindings']['waiver']['sha256']},c['bindings']['dataset']['path'])
    require(not Path(a['output']).exists(),'output already exists; no automatic retry')
    if args.preflight:
        print(json.dumps({'status':'PASS_CPU_PREFLIGHT','cmd':command(a['output'],c),'runtime_limit_seconds':86400}));return
    require(os.environ.get('TMUX'),'main supervisor must run inside tmux')
    attached=subprocess.check_output(['tmux','display-message','-p','#{session_attached}'],text=True).strip()
    require(attached=='0','tmux session must be detached at startup')
    require(Path(sys.executable).resolve()==PYTHON.resolve(),'pinned Python interpreter required')
    lock=open('/data/axonai/engineering-maintenance/.main-sft-gpu.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(not detect_conflict(),'GPU conflict, wrong GPU type/count or ambiguous device state')
    out=prepare(a,c,args.approval,args.approval_sha256)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',FSDP_VERSION='1',AXONAI_APPROVAL_STATUS='APPROVED_MAIN_SFT')
    env.pop('PYTHONPATH',None)
    # Preserve the total runtime budget across the reviewed zero-step failure.
    supervise(command(out,c),env,out,86400-math.ceil(a['prior_attempt_runtime_seconds']))

if __name__=='__main__':main()
