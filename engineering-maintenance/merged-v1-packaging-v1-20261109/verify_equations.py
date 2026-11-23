import json,time,resource,collections,os
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=Path('/data/axonai');BASE=ROOT/'models/base/Qwen3.8-27B';M=ROOT/'models/merged/AxonAI-v1-27B';A=ROOT/'engineering-maintenance/main-sft-run-20261029T150039Z/final_adapter'
from runtime_safeguards import write,restrict_writes

def main():
 (P/"tmp").mkdir(exist_ok=True);os.environ["TMPDIR"]=str(P/"tmp")
 restrict_writes(P)
 import torch
 from safetensors import safe_open
 from safetensors.torch import load_file
 torch.set_num_threads(16);start=time.monotonic();bi=json.loads((BASE/'model.safetensors.index.json').read_text())['weight_map'];mi=json.loads((M/'model.safetensors.index.json').read_text())['weight_map'];ad=load_file(str(A/'adapter_model.safetensors'));cfg=json.loads((A/'adapter_config.json').read_text());assert cfg['lora_alpha']==32 and cfg['r']==16
 bf={n:safe_open(BASE/n,framework='pt',device='cpu') for n in set(bi.values())};mf={n:safe_open(M/n,framework='pt',device='cpu') for n in set(mi.values())};updated=[];copied=[];used=set();errors=[];dtypes=collections.Counter()
 for key,shard in mi.items():
  assert key in bi;original=bf[bi[key]].get_tensor(key);dtypes[str(original.dtype)]+=1;base=original.to(torch.bfloat16);actual=mf[shard].get_tensor(key)
  prefix='base_model.model.'+key.replace('model.language_model.','model.')[:-len('.weight')] if key.endswith('.weight') else None
  ka=prefix+'.lora_A.weight' if prefix else None;kb=prefix+'.lora_B.weight' if prefix else None
  if ka in ad:
   assert kb in ad;expected=(base.float()+(ad[kb]@ad[ka])*2.0).bfloat16();updated.append(key);used.update([ka,kb])
  else:expected=base;copied.append(key)
  if not torch.equal(actual,expected):errors.append(key)
  del original,base,actual,expected
 assert not errors,errors;assert len(updated)==496 and len(used)==992 and used==set(ad)
 write(P/'MERGE_EQUATION_VERIFICATION.json',{'status':'PASS','updated_tensors_exactly_match_fp32_merge_then_bf16':len(updated),'unadapted_tensors_match_loaded_base':len(copied),'all_output_tensors_verified':len(mi),'all_adapter_tensors_consumed':len(used),'source_dtype_counts':dict(dtypes),'base_only_tensors_not_exported':len(set(bi)-set(mi)),'excluded_key_prefix_counts':dict(collections.Counter(k.split('.')[1] if '.' in k else k for k in set(bi)-set(mi))),'equation':'BF16(FP32(BF16(base tensor as loaded)) + 2.0*(FP32(B) @ FP32(A))) for all 496 adapted weights; exact loaded-base equality for remaining 355 tensors','elapsed_seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'inference_bitwise_equivalence_claimed':False})
 print('ALL_851_WEIGHT_EQUATIONS_PASS')
if __name__=='__main__':main()
