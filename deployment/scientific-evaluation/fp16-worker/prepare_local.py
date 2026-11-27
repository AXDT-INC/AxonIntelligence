"""Offline, additive preparation; originals are never changed. Run as a CLI only."""
import pathlib,sys,json,hashlib,shutil,os,time
import argparse
parser=argparse.ArgumentParser(description='Reproduce pinned FP16 inference weights from local BF16 checkpoints.')
parser.add_argument('output',type=pathlib.Path)
args=parser.parse_args()
root=pathlib.Path(__file__).resolve().parents[3];out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);os.umask(0o077)
sys.path.insert(0,str(root/'deployment/fp16-docker'))
from verify import compare_shard,rounding_table
from convert import tensors_for_shard,check_staged
from safetensors.torch import save_file
import torch
torch.set_num_threads(4);torch.set_num_interop_threads(1)
def sha(p):
 with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
src=root/'models/merged/AxonAI-v1-27B'; layout=root/'models/merged/AxonAI-v1-27B-FP16';target=out/'checkpoints/AxonAI-v1-27B-FP16';target.mkdir(parents=True,exist_ok=True)
manifest=json.loads((layout/'SHA256_MANIFEST.json').read_bytes()); rows={x['relative_path']:x for x in manifest['files']}
source_rows={x['relative_path']:x for x in json.loads((src/'SHA256_MANIFEST.json').read_bytes())['files']}
index=json.loads((src/'model.safetensors.index.json').read_bytes())['weight_map'];table=rounding_table();receipt=[]
assert sha(src/'SHA256_MANIFEST.json')=='d34c6389c0ca68df59e11f2a18f3616e7b335066fd8c493acf46044551e1daee'
for name in sorted(set(index.values())):
 start=time.monotonic();original=src/name;final=target/name;keys={k for k,v in index.items() if v==name}
 assert sha(original)==source_rows[name]['sha256']
 if not final.exists():
  values,metadata=tensors_for_shard(original,keys);pending=target/(name+'.part');save_file(values,pending,metadata=metadata);check_staged(pending,values,metadata);del values
  os.rename(pending,final)
 counts,original_sha,checksum=compare_shard(original,final,keys,table)
 assert original_sha==source_rows[name]['sha256'] and checksum==rows[name]['sha256'],(name,checksum,rows[name]['sha256'])
 row={'path':name,'sha256':checksum,'seconds':time.monotonic()-start,'counts':dict(counts)};receipt.append(row);print(json.dumps(row),flush=True)
for name in ['config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','generation_config.json']:
 path=layout/name if (layout/name).exists() else src/name
 assert sha(path)==rows[name]['sha256'];shutil.copy2(path,target/name)
shutil.copy2(layout/'SHA256_MANIFEST.json',target/'SHA256_MANIFEST.json')
(out/'axonai-preparation.json').write_text(json.dumps({'source_manifest':sha(src/'SHA256_MANIFEST.json'),'runtime_manifest':sha(target/'SHA256_MANIFEST.json'),'historical_manifest_runtime_subset':True,'note':'Only inference weights and six runtime support files are materialized. Historical conversion log files referenced in the original manifest are not recreated.','independently_verified_shards':receipt},indent=2)+'\n')
# Reuse the original bare export routine without changing any source checkpoint.
sys.path.insert(0,str(root/'deployment/scientific-evaluation/fp16-worker'))
import prepare_base
prepare_base.SOURCE=root/'models/base/Qwen3.8-27B';prepare_base.LAYOUT=target;prepare_base.OUT=out/'checkpoints';
prepare_base.fixture()
if not (prepare_base.OUT/'Qwen3.8-27B-text-FP16').exists():
 prepare_base.build(root/'models/merged/AxonAI-v1-27B/PROVENANCE.json')
else:
 assert sha(prepare_base.OUT/'Qwen3.8-27B-text-FP16/SHA256_MANIFEST.json')=='644bca802d87721db6656d3ef4f7bf8e8c79bfdbe2761c4abce5086186b06c6e'
 print('Existing pinned Qwen export retained; the worker will verify all runtime files.')
