import os,sys,time,json,resource,traceback,argparse,hashlib,importlib.abc
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=Path('/data/axonai');BASE=ROOT/'models/base/Qwen3.8-27B';ADAPTER=ROOT/'engineering-maintenance/main-sft-run-20261029T150039Z/final_adapter';MERGED=ROOT/'models/merged/AxonAI-v1-27B'
from runtime_safeguards import restrict_writes,write,append

def main(mode,output_name=None):
 out=P/(output_name or mode);out.mkdir(exist_ok=False);write(out/'CONFINEMENT.json',restrict_writes(P))
 if mode!='baseline':
  import importlib.util
  spec=importlib.util.spec_from_file_location('guard_no_peft',P/'no_peft_guard.v2.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard);guard.install()
  def audit(event,args):
   if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
    s=os.fsdecode(args[0]);p=Path(s).resolve()
    if p.is_relative_to(BASE) or p.is_relative_to(ADAPTER):raise PermissionError('Standalone validation forbids original base/adapter reads')
  sys.addaudithook(audit)
 import torch,numpy as np
 from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig,LogitsProcessor,set_seed
 torch.set_num_threads(8);set_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
 assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_name(i)=='NVIDIA L40S' for i in range(2))
 t=time.monotonic();path=BASE if mode=='baseline' else MERGED
 tok=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
 model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa',device_map='balanced',max_memory={0:'38GiB',1:'38GiB'},low_cpu_mem_usage=True)
 assert all(str(v) in {'0','1','cuda:0','cuda:1'} for v in model.hf_device_map.values())
 if mode=='baseline':
  from peft import PeftModel,get_peft_model_state_dict
  from safetensors.torch import load_file
  model=PeftModel.from_pretrained(model,ADAPTER,is_trainable=False,autocast_adapter_dtype=True,local_files_only=True)
  a=load_file(str(ADAPTER/'adapter_model.safetensors'));b=get_peft_model_state_dict(model);assert len(a)==len(b)==992 and all(torch.equal(v,b[k].cpu()) for k,v in a.items());del a,b
 model.requires_grad_(False);model.eval();versions={k:p._version for k,p in model.named_parameters()};loaded=time.monotonic()-t
 prompts=json.loads((P/'VALIDATION_PROMPTS.json').read_text());selected=prompts['prompts'] if mode!='smoke' else [prompts['fresh_smoke']]
 generation=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=384,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,bos_token_id=tok.bos_token_id,use_cache=True)
 class Finite(LogitsProcessor):
  def __call__(self,ids,scores):
   assert torch.isfinite(scores).all(),'Nonfinite logits';return scores
 device=model.get_input_embeddings().weight.device
 for i,task in enumerate(selected):
  set_seed(42);rendered=tok.apply_chat_template(task['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False);inputs=tok(rendered,return_tensors='pt',add_special_tokens=False);n=inputs['input_ids'].shape[1];assert n+384<=4096;inputs={k:v.to(device) for k,v in inputs.items()}
  for d in range(2):torch.cuda.synchronize(d);torch.cuda.reset_peak_memory_stats(d)
  s=time.monotonic()
  with torch.inference_mode():generated=model.generate(**inputs,generation_config=generation,logits_processor=[Finite()])
  for d in range(2):torch.cuda.synchronize(d)
  elapsed=time.monotonic()-s;ids=generated[0,n:].cpu().tolist();text=tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False);assert text.strip()
  r={'id':task['id'],'rendered_sha256':hashlib.sha256(rendered.encode()).hexdigest(),'input_ids':inputs['input_ids'][0].cpu().tolist(),'output_ids':ids,'text':text,'input_tokens':n,'output_tokens':len(ids),'latency_seconds':elapsed,'truncated':len(ids)==384 and ids[-1]!=tok.eos_token_id,'peak_gpu_allocated_bytes':[torch.cuda.max_memory_allocated(d) for d in range(2)]}
  if mode in ['baseline','merged']:
   reference=ids if mode=='baseline' else json.loads((P/'baseline/responses.jsonl').read_text().splitlines()[i])['output_ids']
   k=min(32,len(reference));prefix=reference[:k-1];forced=torch.cat([inputs['input_ids'],torch.tensor([prefix],device=device,dtype=torch.long)],dim=1)
   with torch.inference_mode():logits=model(input_ids=forced,attention_mask=torch.ones_like(forced),use_cache=False,logits_to_keep=k).logits[0].float().cpu().numpy()
   assert logits.shape[0]==k and np.isfinite(logits).all()
   if mode=='baseline':np.save(P/f'baseline/logits-{i}.npy',logits)
   else:
    original=np.load(P/f'baseline/logits-{i}.npy');assert original.shape==logits.shape;diff=logits-original;top=original.argmax(-1);newtop=logits.argmax(-1);r['aligned_baseline_prefix_logits']={'positions':k,'max_abs_difference':float(np.abs(diff).max()),'rmse':float(np.sqrt(np.mean(diff.astype('float64')**2))),'top1_agreement':int((top==newtop).sum()),'baseline_top1':top.tolist(),'merged_top1':newtop.tolist(),'first_token_baseline_top2_margin':float(np.sort(original[0])[-1]-np.sort(original[0])[-2])}
  if mode=='smoke':r['json_smoke_passed']=json.loads(text)==prompts['fresh_smoke']['expected'];assert r['json_smoke_passed']
  append(out/'responses.jsonl',r);print(mode,task['id'],len(ids),round(elapsed,2),flush=True)
 assert all(p._version==versions[k] for k,p in model.named_parameters())
 if mode!='baseline':assert not any(n=='peft' or n.startswith('peft.') for n in sys.modules)
 write(out/'RESULT.json',{'status':'PASS','responses':len(selected),'load_seconds':loaded,'total_seconds':time.monotonic()-t,'peak_cpu_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'model_class':type(model).__name__,'independent_no_peft_load':mode!='baseline','original_paths_reads_denied':mode!='baseline','parameter_versions_unchanged':True,'optimizer_created':False,'device_map':{k:str(v) for k,v in model.hf_device_map.items()}})
 del model
 import gc;gc.collect()
 for d in range(2):torch.cuda.synchronize(d);torch.cuda.empty_cache()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['baseline','merged','smoke']);p.add_argument('--output-name');a=p.parse_args()
 try:main(a.mode,a.output_name)
 except BaseException:(P/((a.output_name or a.mode)+'_TRACEBACK.log')).write_text(traceback.format_exc());raise
