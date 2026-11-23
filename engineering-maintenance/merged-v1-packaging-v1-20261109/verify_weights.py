import json,collections,time
from pathlib import Path
from runtime_safeguards import sha,write,binding
P=Path(__file__).resolve().parent;M=Path('/data/axonai/models/merged/AxonAI-v1-27B')
def verify(root):
 import torch
 from safetensors import safe_open
 torch.set_num_threads(16);start=time.monotonic();index=json.loads((root/'model.safetensors.index.json').read_text());mapping=index['weight_map'];seen=set();total=0;types=collections.Counter();tensors=[];shards=[]
 for name in sorted(set(mapping.values())):
  assert Path(name).name==name and name.endswith('.safetensors');path=root/name
  with safe_open(path,framework='pt',device='cpu') as f:
   for key in f.keys():
    assert key not in seen and mapping[key]==name and 'lora_' not in key
    t=f.get_tensor(key);assert torch.isfinite(t).all(),key;seen.add(key);total+=t.numel()*t.element_size();types[str(t.dtype)]+=1;tensors.append({'name':key,'shape':list(t.shape),'dtype':str(t.dtype),'bytes':t.numel()*t.element_size()});del t
  shards.append(binding(path))
 assert seen==set(mapping) and total==index['metadata']['total_size']
 assert set(x.name for x in root.glob('*.safetensors'))==set(mapping.values())
 assert not list(root.glob('adapter*'))
 return {'status':'PASS','index':binding(root/'model.safetensors.index.json'),'shards':shards,'shard_count':len(shards),'tensor_count':len(seen),'tensor_bytes':total,'shard_file_bytes':sum(x['bytes'] for x in shards),'dtype_counts':dict(types),'all_finite':True,'no_adapter_files':True,'tensors':tensors,'elapsed_seconds':time.monotonic()-start}
if __name__=='__main__':write(P/'WEIGHT_VERIFICATION.json',verify(M));print('WEIGHTS_PASS')
