import sys,json,os,subprocess,shutil,datetime
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=Path('/data/axonai');M=ROOT/'models/merged/AxonAI-v1-27B';BASE=ROOT/'models/base/Qwen3.8-27B'
from runtime_safeguards import binding,sha,write
assert not M.exists() and not (P/'merged-build').exists()
revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0';assert (BASE/'.cache/huggingface/download/config.json.metadata').read_text().splitlines()[0]==revision
inventory=json.loads((ROOT/'engineering-maintenance/mixed-scope-continuation-v1-20261109/MODEL_INTEGRITY.json').read_text());verified=[]
for x in inventory['files']:assert sha(x['path'])==x['sha256'];verified.append(binding(x['path']))
adapter=ROOT/'engineering-maintenance/main-sft-run-20261029T150039Z/final_adapter/adapter_model.safetensors';assert sha(adapter)=='5280cf5f30265038a6fee6aa57955b829d387b7d78fb9fe6afab86d65ac50978'
mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())};assert mem['MemAvailable']>400*1024**3;assert mem['SwapTotal']==0
available=shutil.disk_usage(ROOT).free;assert available>300*1024**3
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True);assert not apps.strip()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True);g=[s.split(', ') for s in gpu.strip().splitlines()];assert all(x[1]=='NVIDIA L40S' and int(x[2])<64 and int(x[3])==0 and int(x[4])==0 for x in g[:2])
cgroup=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().split('::')[1].strip().lstrip('/');limits=[]
for parent in [cgroup,*cgroup.parents]:
 if not parent.is_relative_to('/sys/fs/cgroup'):continue
 p=parent/'memory.max'
 if p.exists():
  v=p.read_text().strip();limits.append({'path':str(p),'value':v});assert v=='max' or int(v)>400*1024**3
now=datetime.datetime.now(datetime.timezone.utc)
write(P/'PREFLIGHT.json',{'utc':now.isoformat(),'verified_source_files':verified,'base_revision':revision,'adapter':binding(adapter),'host_memory':mem,'cgroup_limits':limits,'destination_free_bytes':available,'gpu_csv':gpu,'compute_apps':apps,'budget':{'estimated_cpu_peak_gib':200,'abort_rss_gib':256,'minimum_memavailable_gib':400,'required_disk_free_gib':300,'output_expected_gb_approx':54,'max_cpu_merge_seconds':1800,'max_validation_seconds_per_stage':900,'max_campaign_seconds':5400},'all_passed':True})
write(P/'AUTHORIZATION.json',{'source':'Current user message: AxonAI-v1: Create and Validate Standalone Merged Model','decision_maker':'User, project owner/CTO in conversation','recorded_utc':now.isoformat(),'scope':'One local v1-only merge, bounded comparative validation and fresh standalone smoke, portable packaging; no training, deployment or publication','adapter_sha256':sha(adapter),'base_revision':revision,'v11_excluded':True,'max_campaign_seconds':5400,'stop_before_ist':'2026-11-24T00:00:00+05:30'})
# Six existing permitted prompts: two biomedical methodology controls, one arithmetic, two pinned code cases and one documentation case.
old=ROOT/'engineering-maintenance/post-training-decision-v1-20261109';tasks=[json.loads(x) for x in (old/'REQUEST_MATRIX.jsonl').read_text().splitlines()];ids=['dev-scope-supplement-A-07','dev-scope-supplement-A-08','dev-scope-supplement-F-07','axonai20k-fe1eb322ad348273c4018583','axonai20k-8bd2585d6ffe6796cd30a174','v11-heldout-WP_VISION']
selected=[next(x for x in tasks if x['id']==id and x['arm']=='B') for id in ids]
write(P/'VALIDATION_PROMPTS.json',{'source':binding(old/'REQUEST_MATRIX.jsonl'),'prompts':selected,'decoding':{'do_sample':False,'num_beams':1,'max_new_tokens':384,'enable_thinking':False,'seed':42,'attention':'sdpa','dtype':'bfloat16','total_token_limit':4096},'fresh_smoke':{'id':'standalone-smoke-v154','messages':[{'role':'system','content':'Follow the requested output format exactly.'},{'role':'user','content':'Return only valid JSON with exactly these two fields: "status": "ok", "sum": the result of 7 + 5.'}],'expected':{'status':'ok','sum':12}},'comparison':'Packaging equivalence on exposed development material, not a capability benchmark.'})
print('PREFLIGHT_PASS')
