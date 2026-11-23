#!/usr/bin/env python3
"""Offline text-only inference. No PEFT, base-model path, adapter or RAG required."""
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--prompt',required=True);p.add_argument('--max-new-tokens',type=int,default=256);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
 if not 1<=a.max_new_tokens<=384:p.error('--max-new-tokens must be between 1 and 384 for the validated envelope')
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig,set_seed
 if torch.cuda.device_count()!=2:raise RuntimeError('Validated example requires exactly two visible GPUs; set CUDA_VISIBLE_DEVICES accordingly')
 set_seed(a.seed);torch.backends.cuda.matmul.allow_tf32=False
 model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa',device_map='balanced',max_memory={0:'38GiB',1:'38GiB'})
 if not all(str(v) in {'0','1','cuda:0','cuda:1'} for v in model.hf_device_map.values()):raise RuntimeError('CPU/disk offload is outside this validated example')
 model.requires_grad_(False);model.eval();tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False)
 messages=[{'role':'system','content':'You are AxonAI, an experimental scientific research assistant. State uncertainty and distinguish supplied evidence from inference.'},{'role':'user','content':a.prompt}]
 text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False);inputs=tok(text,add_special_tokens=False,return_tensors='pt');n=inputs['input_ids'].shape[-1]
 if n+a.max_new_tokens>4096:raise ValueError('Input plus output reserve exceeds validated 4096-token limit; no truncation performed')
 inputs={k:v.to(model.get_input_embeddings().weight.device) for k,v in inputs.items()};g=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=a.max_new_tokens,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,bos_token_id=tok.bos_token_id,use_cache=True)
 with torch.inference_mode():out=model.generate(**inputs,generation_config=g)
 if not torch.isfinite(model.get_output_embeddings().weight).all():raise FloatingPointError('Nonfinite output-head weight')
 print(tok.decode(out[0,n:],skip_special_tokens=True,clean_up_tokenization_spaces=False))
if __name__=='__main__':main()
