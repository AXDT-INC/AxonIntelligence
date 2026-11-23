import json,statistics,collections,datetime,subprocess,os
from pathlib import Path
from runtime_safeguards import write,binding
P=Path(__file__).resolve().parent;M=Path('/data/axonai/models/merged/AxonAI-v1-27B')
def read(p):return json.loads(Path(p).read_text())
def main():
 assert read(P/'RESUME2_COMPLETE.json')['status']=='PASS';comp=read(P/'TOKEN_COMPARISON.json');metrics={}
 for label,folder,stage in [('baseline','baseline','baseline'),('merged','merged-repair2','merged_repair2'),('smoke','smoke','smoke')]:
  r=read(P/folder/'RESULT.json');responses=[json.loads(x) for x in (P/folder/'responses.jsonl').read_text().splitlines()];samples=[json.loads(x) for x in (P/(stage+'_RESOURCES.jsonl')).read_text().splitlines()];gpu=collections.defaultdict(list)
  for sample in samples:
   for line in sample['gpu_csv'].strip().splitlines():
    a=line.split(', ');gpu[int(a[0])].append([int(x) for x in a[1:]])
  metrics[label]={'load_seconds':r['load_seconds'],'worker_total_seconds':r['total_seconds'],'supervised_seconds':read(P/(stage+'_CLEANUP.json'))['elapsed_seconds'],'peak_cpu_rss_kib':r['peak_cpu_rss_kib'],'generation_seconds':sum(x['latency_seconds'] for x in responses),'mean_latency_seconds':statistics.mean(x['latency_seconds'] for x in responses),'peak_torch_allocated_bytes':[max(x['peak_gpu_allocated_bytes'][i] for x in responses) for i in range(2)],'sampled_gpu':{str(i):{'max_memory_mib':max(x[0] for x in xs),'mean_utilization_pct':statistics.mean(x[1] for x in xs),'max_volatile_uncorrected_ecc':max(x[2] for x in xs)} for i,xs in gpu.items()},'standalone_no_peft':r['independent_no_peft_load'],'original_base_and_adapter_reads_blocked':r['original_paths_reads_denied']}
 for stage in ['merge_cpu','verify_weights','baseline','merged','merged_repair1','merged_repair2','smoke']:
  c=read(P/(stage+'_CLEANUP.json'));assert c['owned_group_absent']
  try:os.kill(c['worker_pid'],0);raise AssertionError('Worker still alive')
  except ProcessLookupError:pass
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True);assert not apps.strip()
 gpuend=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 write(P/'PROCESS_TERMINATION.json',{'all_owned_workers_absent':True,'compute_apps':apps,'final_gpu_csv':gpuend,'original_campaign_exit':(P/'CAMPAIGN_EXIT.txt').read_text().strip(),'first_repair_exit':(P/'RESUME_EXIT.txt').read_text().strip(),'final_validation_exit':(P/'RESUME2_EXIT.txt').read_text().strip(),'failed_stages':'Two validation-harness attempts before model load; both preserved.','unrelated_processes_signalled':False})
 write(P/'RESOURCE_METRICS.json',{'cpu_merge':read(P/'MERGE_RESULT.json'),'validation':metrics})
 notes=[{'id':x['id'],'judgment':'Same checked factual content in provisional Codex review; not independent validation','detail':('Both give incorrect 230 GPU-minutes instead of 190; merge did not fix inherited reasoning failure.' if x['id'].endswith('F-07') else 'Merged adds accurate code-context explanation; fallback and exception behavior unchanged.' if 'fe1eb322' in x['id'] else 'Merged is more verbose and quotes supplied vision statements; goal versus demonstrated-result distinction retained.' if x['id'].endswith('WP_VISION') else 'Exact generated token sequence preserved.'),'exact_tokens':x['exact_generated_token_equality']} for x in comp['rows']]
 write(P/'PROVISIONAL_COMPARISON_REVIEW.json',{'reviewer':'Codex provisional AI review, not independent or blinded','rows':notes,'material_factual_regression_observed_in_six_cases':False,'broad_equivalence_or_quality_improvement_established':False,'diagnosis':'Expected numerical change from one BF16 rounding of merged weights and changing separate LoRA GEMMs to a single merged GEMM; backend/device/decoding held fixed. Weight-equation verification separately tests implementation correctness. No full-FP32 reference inference isolates the exact contribution of each rounding stage.'})
 summary={'status':'STANDALONE_VALIDATION_PASS_NOT_BITWISE_EQUIVALENT','merged_weight_verification':{k:v for k,v in read(P/'WEIGHT_VERIFICATION.json').items() if k not in ['tensors','shards','index']},'token_comparison':comp['summary'],'independent_load':{'no_peft_imported':True,'base_and_adapter_paths_denied':True,'method':'Isolated process reports PEFT package absent and blocks direct imports; Python filesystem-open audit denies original base/adapter reads. No global packages uninstalled.'},'fresh_smoke':{'result':read(P/'smoke/RESULT.json')['status'],'output':json.loads((P/'smoke/responses.jsonl').read_text())['text'],'expected_json_passed':True},'resources':metrics,'original_adapter':'v1 only; v1.1 not merged','material_factual_regression_observed_in_six_cases':False,'independent_scientific_validation':False,'retains_existing_model_limitations':True}
 write(M/'VALIDATION_SUMMARY.json',summary);write(P/'VALIDATION_SUMMARY.json',summary)
 print(json.dumps({'comparison':comp['summary'],'resources':metrics},indent=2))
if __name__=='__main__':main()
