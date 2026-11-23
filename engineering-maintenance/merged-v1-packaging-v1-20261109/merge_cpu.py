import os,time,json,resource,collections,traceback
from pathlib import Path
P=Path(__file__).resolve().parent;STAGE=P/'merged-build';BASE=Path('/data/axonai/models/base/Qwen3.8-27B');ADAPTER=Path('/data/axonai/engineering-maintenance/main-sft-run-20261029T150039Z/final_adapter')
from runtime_safeguards import restrict_writes,write,binding

def main():
 write(P/'MERGE_CONFINEMENT.json',restrict_writes(P))
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from peft import PeftModel,get_peft_model_state_dict
 from safetensors.torch import load_file
 torch.set_num_threads(16);start=time.monotonic()
 model=AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map={'':'cpu'},low_cpu_mem_usage=True,attn_implementation='sdpa')
 loaded=time.monotonic();original_dtypes={n:p.dtype for n,p in model.named_parameters()};original_buffers={n:b.dtype for n,b in model.named_buffers()};counts=collections.Counter(str(p.dtype) for p in model.parameters())
 model=model.float();model=PeftModel.from_pretrained(model,ADAPTER,is_trainable=False,autocast_adapter_dtype=True,local_files_only=True)
 disk=load_file(str(ADAPTER/'adapter_model.safetensors'));loaded_adapter=get_peft_model_state_dict(model)
 assert len(disk)==len(loaded_adapter)==992 and set(disk)==set(loaded_adapter)
 assert all(t.dtype==torch.float32 and torch.equal(t,loaded_adapter[k].cpu()) for k,t in disk.items());del disk,loaded_adapter
 modules=[(n,m) for n,m in model.named_modules() if hasattr(m,'lora_A') and 'default' in m.lora_A];assert len(modules)==496
 checks=[]
 for n,m in [modules[0],modules[len(modules)//2],modules[-1]]:
  checks.append((n,m.get_base_layer(),m.get_base_layer().weight.detach().clone()+m.get_delta_weight('default')))
 merged=model.merge_and_unload(safe_merge=True);assert not any('lora_' in k for k in merged.state_dict())
 for n,m,expected in checks:assert torch.equal(m.weight,expected),n
 del checks,modules,model
 assert set(original_dtypes)=={n for n,p in merged.named_parameters()}
 for n,p in merged.named_parameters():
  assert torch.isfinite(p).all(),n
  p.data=p.data.to(original_dtypes[n])
 for n,b in merged.named_buffers():
  if n in original_buffers and b.is_floating_point():b.data=b.data.to(original_buffers[n])
 merged.requires_grad_(False);merged.eval();assert not any(p.requires_grad for p in merged.parameters())
 merge_done=time.monotonic();STAGE.mkdir(exist_ok=False)
 merged.save_pretrained(STAGE,safe_serialization=True,max_shard_size='4GB')
 tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True,trust_remote_code=False);tok.save_pretrained(STAGE)
 write(P/'MERGE_RESULT.json',{'status':'PASS','model_class':type(merged).__name__,'config_class':type(merged.config).__name__,'base_load_seconds':loaded-start,'merge_and_conversion_seconds':merge_done-loaded,'save_seconds':time.monotonic()-merge_done,'total_seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'original_parameter_dtype_counts':dict(counts),'output_parameter_dtype_counts':dict(collections.Counter(str(p.dtype) for p in merged.parameters())),'parameter_count':sum(p.numel() for p in merged.parameters()),'adapter_tensors_exact':992,'merged_modules':496,'safe_merge':True,'fp32_arithmetic_spot_checks':3,'procedure':'CPU BF16 load preserving loader FP32 exceptions; promote to FP32; attach exact FP32 v1 LoRA; PEFT safe_merge; restore original per-parameter dtypes once; safe sharded serialization','precision_limitation':'BF16 output rounds W+scaled(B@A) once. Separate adapter GEMMs and dtype conversions can yield different outputs; no bitwise equivalence assumed.','text_only':True,'optimizer_created':False})
 print('MERGE_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:(P/'MERGE_TRACEBACK.log').write_text(traceback.format_exc());raise
