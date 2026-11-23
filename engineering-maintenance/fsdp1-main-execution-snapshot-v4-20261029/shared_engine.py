import argparse,hashlib,json,os,signal,subprocess,time
DATASET_SHA256='b0810d15a4aa2968b03690a63a6d38d0ba9f9a9162acd357d08f77dbe772c8a4'; CONFIG_SHA256='b5b7339929e29483b6251e1727affdd63664e3a310dafe40d512a46d67fad018'; MODEL_REVISION='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0';
def checkpoint_contract():
 return {'backend':'torch.distributed.checkpoint','required_state':['model','optimizer','scheduler','rng','global_step'],'isolated_output':True}
EXPECTED_CONFIG={'world_size':8,'max_length':4096,'num_train_epochs':2,'per_device_train_batch_size':1,'gradient_accumulation_steps':4,'learning_rate':2e-05,'weight_decay':0.01,'warmup_ratio':0.03,'lr_scheduler_type':'cosine','save_steps':250,'checkpoint_minutes':15}
def validate_config(cfg):
 for k,v in EXPECTED_CONFIG.items():
  if cfg.get(k)!=v: raise ValueError(f'config mismatch: {k}')
 if cfg.get('expected_model_revision')!=MODEL_REVISION: raise ValueError('model revision mismatch')
 return True
def tokenize_record(record, tokenizer, max_length=4096):
 if 'input_ids' in record and 'labels' in record: return record
 messages=record.get('messages') or [{'role':'user','content':record.get('question','')},{'role':'assistant','content':record.get('answer','')}]
 text='\n'.join(str(m.get('content','')) for m in messages)
 enc=tokenizer(text,truncation=True,max_length=max_length); ids=enc['input_ids']
 return {'input_ids':ids,'labels':ids.copy(),'record_id':record.get('record_id')}
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def authorize(receipt,mode,dataset,extraction_receipt=None):
 r=json.load(open(receipt)); expected='GPU_SMOKE_APPROVED' if mode=='SMOKE' else 'APPROVED_MAIN_SFT'
 if r.get('status')!=expected or r.get('dataset_sha256')!=DATASET_SHA256: raise RuntimeError('authorization mismatch')
 if mode=='SMOKE':
  ep=extraction_receipt or os.environ.get('AXONAI_EXTRACTION_RECEIPT')
  if not ep: raise RuntimeError('extraction receipt required')
  e=json.load(open(ep))
  if e.get('status')!='COMPLETE' or e.get('approved_manifest_sha256')!=DATASET_SHA256 or e.get('extracted_jsonl_sha256')!=sha(dataset) or e.get('record_count')!=10684 or not e.get('id_coverage_exact'): raise RuntimeError('dataset extraction binding mismatch')
def fsdp2_wrap(model):
 from torch.distributed._composable.fsdp import fully_shard
 layers=[]
 for name,module in model.named_modules():
  if module is not model and module.__class__.__name__ in ('Qwen3_5DecoderLayer','Qwen3DecoderLayer','Qwen2DecoderLayer'):
   layers.append((name,module))
 if not layers: raise RuntimeError('no supported decoder layers found')
 for _,layer in layers: fully_shard(layer)
 fully_shard(model)
 return model
def load_lora_model(model_path,cfg):
 import torch
 from transformers import AutoConfig,AutoModelForCausalLM
 from peft import LoraConfig,get_peft_model
 if not os.path.isdir(model_path): raise RuntimeError('model path missing')
 try:
  from accelerate import init_empty_weights,load_checkpoint_and_dispatch
 except Exception as e: raise RuntimeError('accelerate meta loading unavailable') from e
 config=AutoConfig.from_pretrained(model_path,revision=MODEL_REVISION,local_files_only=True)
 with init_empty_weights(): model=AutoModelForCausalLM.from_config(config,torch_dtype=torch.bfloat16)
 # Dispatch streams safetensors into the meta model; no blanket GPU transfer.
 model=load_checkpoint_and_dispatch(model,model_path,device_map={'':'cpu'},offload_state_dict=True)
 if any(getattr(p,'is_meta',False) for p in model.parameters()): raise RuntimeError('unmaterialized pretrained parameter')
 lc=cfg['lora']; return get_peft_model(model,LoraConfig(r=lc['r'],lora_alpha=lc['alpha'],lora_dropout=lc['dropout'],target_modules=lc['target_modules'],task_type='CAUSAL_LM'))
def checkpoint_save(state,path):
 import torch.distributed.checkpoint as dcp
 staging=path+'.staging'; os.makedirs(staging,exist_ok=False); dcp.save(state,storage_writer=dcp.FileSystemWriter(staging)); os.replace(staging,path)
def checkpoint_load(state,path):
 import torch.distributed.checkpoint as dcp
 if not os.path.isdir(path) or os.path.exists(path+'.staging'): raise RuntimeError('incomplete checkpoint')
 dcp.load(state,storage_reader=dcp.FileSystemReader(path))
def checkpoint_resume(state,path,metadata):
 if not os.path.isfile(os.path.join(path,'metadata.json')): raise RuntimeError('checkpoint metadata missing')
 saved=json.load(open(os.path.join(path,'metadata.json')))
 if saved!=metadata: raise RuntimeError('checkpoint identity mismatch')
 checkpoint_load(state,path); return state
def publish_checkpoint(state,path,metadata):
 import torch.distributed as dist
 if dist.is_available() and dist.is_initialized(): dist.barrier()
 staged=path+'.staging'; os.makedirs(staged,exist_ok=False); json.dump(metadata,open(os.path.join(staged,'metadata.json'),'w'),sort_keys=True)
 import torch.distributed.checkpoint as dcp
 dcp.save(state,storage_writer=dcp.FileSystemWriter(staged))
 if dist.is_available() and dist.is_initialized(): dist.barrier()
 if dist.get_rank()==0 if dist.is_available() and dist.is_initialized() else True: os.replace(staged,path)
 if dist.is_available() and dist.is_initialized(): dist.barrier()
def detect_conflict():
 try:
  visible=subprocess.check_output(['nvidia-smi','--query-gpu=index,name','--format=csv,noheader'],text=True,stderr=subprocess.STDOUT,timeout=10)
  rows=[x.strip() for x in visible.splitlines() if x.strip()]
  if len(rows)!=8 or any('NVIDIA L40S' not in x for x in rows): return True
  out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True,stderr=subprocess.STDOUT,timeout=10).strip()
  if not out or out.lower() in ('no running processes found','no running processes'): return False
  for line in out.splitlines():
   parts=[p.strip() for p in line.split(',')]
   if len(parts)!=3 or not parts[0].isdigit() or not parts[1] or parts[2].lower() in ('n/a','[not supported]'): return True
  return True
 except Exception:return True
def cleanup(proc,timeout=15):
 try:
  import psutil; p=psutil.Process(proc.pid); ps=[p]+p.children(recursive=True)
  for x in ps:x.terminate()
  gone,_=psutil.wait_procs(ps,timeout=timeout)
  for x in ps:
   if x.is_running(): x.kill()
  return [x.pid for x in ps if x.is_running()]
 except Exception:return [proc.pid] if proc.poll() is None else []
def supervise_process(cmd, mode, timeout, env=None, approval=None, dataset=None, config=None, output=None, extraction_receipt=None):
 expected={'SMOKE':'GPU_SMOKE_APPROVED','MAIN_SFT':'APPROVED_MAIN_SFT','CHECKPOINT_RESUME_TEST':'CHECKPOINT_RESUME_TEST'}.get(mode)
 if not expected: raise ValueError('invalid mode')
 if not env or env.get('AXONAI_APPROVAL_STATUS')!=expected or not approval: raise RuntimeError('authorization mismatch')
 r=json.load(open(approval));
 if not config or not os.path.isfile(config): raise RuntimeError('selected config missing')
 cfg_hash=sha(config); cfg=json.load(open(config))
 expected_cfg={'SMOKE':'smoke','MAIN_SFT':'main','CHECKPOINT_RESUME_TEST':'resume'}[mode]
 if r.get('status')!=expected or r.get('dataset_sha256')!=DATASET_SHA256 or r.get('config_sha256')!=cfg_hash or cfg.get('mode')!=expected_cfg: raise RuntimeError('approval binding mismatch')
 if mode=='SMOKE':
  if extraction_receipt is None or not os.path.isfile(extraction_receipt): raise RuntimeError('extraction receipt required')
  e=json.load(open(extraction_receipt))
  if e.get('status')!='COMPLETE' or e.get('approved_manifest_sha256')!=DATASET_SHA256 or e.get('extracted_jsonl_sha256')!=sha(dataset) or e.get('record_count')!=10684 or not e.get('id_coverage_exact'):
   raise RuntimeError('dataset extraction binding mismatch')
 if detect_conflict(): raise RuntimeError('GPU conflict or ambiguous state')
 p=subprocess.Popen(cmd,env=env,start_new_session=True)
 try:
  rc=p.wait(timeout=timeout)
  if rc!=0: raise RuntimeError(f'worker failure: {rc}')
  return {'status':'PROCESS_COMPLETE','returncode':rc,'survivors':[]}
 except Exception as exc:
  survivors=cleanup(p,timeout=min(15,timeout));
  if survivors: raise RuntimeError(f'worker survivors: {survivors}')
  raise RuntimeError('supervised worker failed or timed out') from exc
def launch_supervised(mode,dataset,config,approval,output,env=None):
 timeout=900 if mode=='SMOKE' else 86400
 child_env=dict(os.environ,**(env or {})); child_env.update({'AXONAI_SUPERVISED':'1','AXONAI_APPROVAL_STATUS':json.load(open(approval)).get('status',''),'AXONAI_CONFIG_PATH':config,'AXONAI_MODEL_PATH':json.load(open(config)).get('model_path','')})
 os.makedirs(output,exist_ok=False)
 cmd=['torchrun','--standalone','--nproc_per_node=8','-m','shared_engine','--worker','--mode',mode,'--dataset',dataset,'--config',config,'--approval',approval,'--output',output]
 return supervise_process(cmd,mode,timeout,child_env,approval,dataset,config,output)
def checkpoint_roundtrip_cpu(state,path,metadata):
 import torch
 os.makedirs(path,exist_ok=True); meta=os.path.join(path,'metadata.json'); json.dump(metadata,open(meta,'w'),sort_keys=True)
 state=dict(state); state['metadata']=metadata; torch.save(state,os.path.join(path,'state.pt')); loaded=torch.load(os.path.join(path,'state.pt'),map_location='cpu')
 required={'model','optimizer','scheduler','rng','global_step','epoch','sampler_progress'}
 if not required.issubset(loaded): raise ValueError('incomplete checkpoint')
 if loaded.get('metadata',metadata)!=metadata: raise ValueError('checkpoint metadata mismatch')
 return loaded
def run_cpu_training(rows, output, cfg=None):
 import torch
 cfg=cfg or {'num_train_epochs':1,'gradient_accumulation_steps':1,'learning_rate':1e-3,'save_steps':250}
 model=torch.nn.Linear(4,2); opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate']); total=max(1,len(rows)*cfg['num_train_epochs']//max(1,cfg['gradient_accumulation_steps'])); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=total); step=0; micro=0; last_loss=None
 for epoch in range(cfg['num_train_epochs']):
  for row in rows:
   x=torch.tensor(row['x'],dtype=torch.float32).reshape(1,-1); raw=row['y']; y=torch.tensor([int(raw if isinstance(raw,int) else raw.index(max(raw)))],dtype=torch.long); loss=torch.nn.functional.cross_entropy(model(x),y)/cfg['gradient_accumulation_steps']; loss.backward(); micro+=1; last_loss=loss
   if micro % cfg['gradient_accumulation_steps']==0:
    opt.step(); sch.step(); opt.zero_grad(); step+=1
 ck=os.path.join(output,'cpu-checkpoint.pt'); rng=torch.get_rng_state(); torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'step':step,'micro_steps':micro,'rng':rng},ck)
 restored=torch.load(ck,map_location='cpu'); model.load_state_dict(restored['model']); opt.load_state_dict(restored['optimizer']); sch.load_state_dict(restored['scheduler']); torch.set_rng_state(restored['rng']); rng_restored=bool(torch.equal(torch.get_rng_state(),restored['rng']))
 return {'steps':step,'micro_steps':micro,'loss':float(last_loss.detach()),'checkpoint':ck,'restored_step':restored['step'],'rng_restored':rng_restored}
def execute_training(mode, dataset, output, cpu_test=False, sample_limit=2):
 if mode not in ('SMOKE','MAIN_SFT'): raise ValueError('invalid mode')
 if not os.path.isdir(output): os.makedirs(output,exist_ok=False)
 if cpu_test:
  rows=load_rows(dataset,sample_limit); result=run_cpu_training(rows,output); result.update({'mode':mode,'status':'CPU_EXECUTION_COMPLETE'}); return result
 return run_distributed_training(mode,dataset,output)
def run_distributed_training(mode,dataset,output):
 import torch
 if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; fail closed')
 local_rank=int(os.environ.get('LOCAL_RANK','0')); world=int(os.environ.get('WORLD_SIZE','1'))
 if world != 8: raise RuntimeError('world size must be exactly 8')
 torch.cuda.set_device(local_rank)
 import torch.distributed as dist
 dist.init_process_group('nccl')
 try:
  # The production model path is deliberately loaded only after distributed setup.
  model_path=os.environ.get('AXONAI_MODEL_PATH')
  config_path=os.environ.get('AXONAI_CONFIG_PATH')
  if not model_path or not config_path: raise RuntimeError('model/config path required')
  cfg=json.load(open(config_path)); validate_config(cfg); model=load_lora_model(model_path,cfg)
  from transformers import AutoTokenizer
  tokenizer=AutoTokenizer.from_pretrained(model_path,revision=MODEL_REVISION,local_files_only=True)
  model=fsdp2_wrap(model)
  rows=load_rows(dataset,2 if mode=='SMOKE' else None); opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=float(cfg['learning_rate']),weight_decay=float(cfg['weight_decay']))
  total=max(1,len(rows)*int(cfg['num_train_epochs'])//int(cfg['gradient_accumulation_steps'])); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=total); model.train(); start=time.monotonic(); steps=0; micro=0; tokens=0
  metadata={'dataset_sha256':DATASET_SHA256,'config_sha256':CONFIG_SHA256,'model_revision':MODEL_REVISION}
  resume=os.environ.get('AXONAI_RESUME_CHECKPOINT')
  if resume:
   checkpoint_resume({'model':model,'optimizer':opt,'scheduler':sch,'global_step':steps,'micro_step':micro},resume,metadata)
  def save_runtime(tag):
   state={'model':model,'optimizer':opt,'scheduler':sch,'global_step':steps,'epoch':0,'sampler_progress':micro,'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all()}
   publish_checkpoint(state,os.path.join(output,tag),metadata)
  for epoch in range(int(cfg['num_train_epochs'])):
   for row in rows:
    if time.monotonic()-start >= (900 if mode=='SMOKE' else 86400): raise TimeoutError('training deadline exceeded')
    tok=tokenize_record(row,tokenizer,int(cfg['max_length'])); ids=torch.tensor(tok['input_ids'],device='cuda',dtype=torch.long).unsqueeze(0); labels=torch.tensor(tok['labels'],device='cuda',dtype=torch.long).unsqueeze(0); tokens += ids.numel()
    out=model(input_ids=ids,labels=labels); loss=out.loss
    if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
    (loss/int(cfg['gradient_accumulation_steps'])).backward(); micro+=1
    if micro % int(cfg['gradient_accumulation_steps'])==0:
     opt.step(); sch.step(); opt.zero_grad(); steps+=1
     if steps % int(cfg['save_steps'])==0: dist.barrier(); save_runtime(f'checkpoint-{steps}')
  if micro % int(cfg['gradient_accumulation_steps']): opt.step(); sch.step(); opt.zero_grad(); steps+=1
  dist.barrier(); save_runtime('checkpoint-final')
  result={'mode':mode,'rank':dist.get_rank(),'world_size':world,'steps':steps,'micro_steps':micro,'tokens':tokens,'status':'GPU_EXECUTION_COMPLETE'}
  if dist.get_rank()==0: json.dump(result,open(os.path.join(output,'RESULT.json'),'w'),sort_keys=True)
  return result
 finally:
  if dist.is_initialized(): dist.destroy_process_group()
def load_rows(path, limit=None):
 rows=[]
 with open(path) as f:
  for line in f:
   if line.strip(): rows.append(json.loads(line))
   if limit is not None and len(rows)>=limit: break
 return rows
def main(mode=None):
 ap=argparse.ArgumentParser()
 if mode is None: ap.add_argument('--mode',choices=['SMOKE','MAIN_SFT'],required=True)
 ap.add_argument('--worker',action='store_true')
 ap.add_argument('--dataset',required=True); ap.add_argument('--config',required=True)
 ap.add_argument('--approval',required=True); ap.add_argument('--output',required=True); ap.add_argument('--cpu-test',action='store_true')
 a=ap.parse_args(); a.mode=mode or a.mode
 if a.worker and os.environ.get('AXONAI_SUPERVISED')!='1': raise SystemExit('direct worker invocation rejected')
 if sha(a.config)!=CONFIG_SHA256: raise SystemExit('config hash mismatch')
 authorize(a.approval,a.mode,a.dataset)
 if detect_conflict(): raise SystemExit('GPU conflict or ambiguous nvidia-smi state')
 if a.worker: result=execute_training(a.mode,a.dataset,a.output,cpu_test=False)
 elif a.cpu_test: result=execute_training(a.mode,a.dataset,a.output,cpu_test=True)
 else: result=launch_supervised(a.mode,a.dataset,a.config,a.approval,a.output)
 print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
