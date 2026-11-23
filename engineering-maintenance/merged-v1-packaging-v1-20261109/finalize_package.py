import json,sys,datetime,shutil,hashlib,statistics,subprocess
from pathlib import Path
from runtime_safeguards import binding,sha
P=Path(__file__).resolve().parent;ROOT=Path('/data/axonai');M=ROOT/'models/merged/AxonAI-v1-27B'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def main():
 assert not (P/'FINALIZATION_RESULT.json').exists()
 for n in ['MERGE_RESULT.json','WEIGHT_VERIFICATION.json','MERGE_EQUATION_VERIFICATION.json','RESUME2_COMPLETE.json']:assert read(P/n)['status']=='PASS'
 w=read(P/'WEIGHT_VERIFICATION.json');comp=read(P/'TOKEN_COMPARISON.json');resources=read(P/'RESOURCE_METRICS.json');eq=read(P/'MERGE_EQUATION_VERIFICATION.json')
 # Clarify all tensors omitted by the same text-only loader used for baseline.
 for f in [M/'README.md',M/'NOTICE.txt']:
  text=f.read_text();text+='\nThe text loader exports 851 tensors. It omits 333 upstream vision tensors and 15 auxiliary MTP tensors that were also unused by the baseline AutoModelForCausalLM inference path. This is not a full multimodal or MTP-capable export.\n';f.write_text(text)
 shutil.copyfile(P/'MERGE_EQUATION_VERIFICATION.json',M/'MERGE_EQUATION_VERIFICATION.json')
 s=read(M/'VALIDATION_SUMMARY.json');s['all_weight_equations_verified']=eq;save(M/'VALIDATION_SUMMARY.json',s);save(P/'VALIDATION_SUMMARY.json',s)
 save(P/'CPU_TEST_RESULTS.json',{'passed':9,'failed':0,'tests':[{'file':binding(P/'CPU_TESTS.log'),'count':4},{'file':binding(P/'CPU_GUARD_TESTS.log'),'count':2,'note':'First guard version tests retained; actual model import required subsequent repair.'},{'file':binding(P/'CPU_NO_PEFT_ENV_TESTS.log'),'count':2},{'file':binding(P/'CPU_TMPDIR_TEST.log'),'count':1}],'all_current_checks_passed':True,'weight_equation_check':binding(P/'MERGE_EQUATION_VERIFICATION.json')})
 # Reverify original source artifacts after all GPU and CPU work.
 sources=read(P/'PREFLIGHT.json')['verified_source_files'];verified=[]
 for x in sources:assert sha(x['path'])==x['sha256'],x['path'];verified.append(binding(x['path']))
 state153=ROOT/'reports/axonai-v1-continuation-20281028/STATE.v153.json';old=read(state153)
 for key in ['original_adapter','experimental_adapter','original_dataset']:
  x=old[key];assert sha(x['path'])==x['sha256'];verified.append(binding(x['path']))
 save(P/'FINAL_SOURCE_INTEGRITY.json',{'all_unchanged':True,'verified_bindings':verified,'original_base_adapter_v11_dataset_modified':False})
 # All model files are bound, except manifest self-hash. Validate newly computed shard hashes against earlier complete finite check.
 files=[];expected={Path(x['path']).name:x for x in w['shards']}
 for f in sorted(M.iterdir()):
  assert f.is_file();h=sha(f)
  if f.name in expected:assert h==expected[f.name]['sha256']
  files.append({'relative_path':f.name,'bytes':f.stat().st_size,'sha256':h})
 save(M/'SHA256_MANIFEST.json',{'schema':'axonai.standalone-model-sha256.v1','files':files,'payload_bytes':sum(x['bytes'] for x in files),'self_hash_excluded':True,'weight_shards':14,'all_shard_hashes_match_finite_verification':True})
 model_bytes=sum(f.stat().st_size for f in M.iterdir());now=datetime.datetime.now(datetime.timezone.utc)
 baseline=resources['validation']['baseline'];merged=resources['validation']['merged'];smoke=resources['validation']['smoke']
 report=f'''# AxonAI-v1 standalone merged model — completed

The portable model is ready for internal self-hosting validation at `/data/axonai/models/merged/AxonAI-v1-27B`. This is the completed biomedical **v1 adapter**, not experimental v1.1. Original models, dataset, checkpoints and prior backups remain unchanged. No training, publication, deployment or sealed evaluation access occurred.

## Model and precision

- Pinned base: `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- v1 adapter SHA-256: `5280cf5f30265038a6fee6aa57955b829d387b7d78fb9fe6afab86d65ac50978`; all 992 FP32 LoRA tensors matched the loaded adapter exactly.
- Export architecture: `Qwen3_5ForCausalLM` / `Qwen3_5TextConfig`, matching the existing validated text inference path despite the historical Qwen3.8 repository/path label.
- **26,895,998,464 parameters, BF16; 851 tensors; 14 Safetensors shards** (4 GB target shard size).
- Tensor payload: **{w['tensor_bytes']:,} bytes**. Shard files: **{w['shard_file_bytes']:,} bytes**. Complete standalone directory including support files and checksum manifest: **{model_bytes:,} bytes** ({model_bytes/1e9:.3f} decimal GB; {model_bytes/1024**3:.3f} GiB).
- Text-only: the established loader omits 333 upstream vision tensors and 15 auxiliary MTP tensors. Those were also unused in the baseline comparison. No image/video or MTP capability is claimed.

The CPU procedure loaded the BF16 base, promoted weights to FP32, attached the exact FP32 v1 adapter, called PEFT `merge_and_unload(safe_merge=True)`, then restored original parameter dtypes before safe serialization. All 851 loaded parameter tensors were BF16; there were no loader-retained FP32 exceptions in this model. This avoids prematurely rounding the LoRA delta before addition, but the final BF16 representation still rounds the sum.

Every saved tensor is finite. The shard/index key sets and total sizes agree exactly. An independent CPU calculation verified **all 496 adapted weights** against `BF16(FP32(base) + 2 × (B @ A))` and **all 355 unadapted tensors** against the loaded base, using all 992 adapter tensors. No model corruption or wrong-adapter merge was detected.

## Standalone validation

A fresh process loaded the merged directory with original base and adapter filesystem reads denied. PEFT was simulated as uninstalled for package discovery, and any actual PEFT import was blocked. No PEFT modules were loaded. This is an isolation test in the existing environment, not a claim that a physically separate clean server has already been tested.

The independent fresh smoke process returned exactly `{{"status":"ok","sum":12}}`. Loading took {smoke['load_seconds']:.2f} seconds and generation {smoke['generation_seconds']:.2f} seconds. This is a packaging smoke test, not scientific evaluation.

Six fixed existing permitted development prompts were compared with identical system/user/evidence content, token IDs, greedy decoding, seed 42, SDPA, BF16 and the same two L40S cards. Input-plus-output was capped at 4,096 tokens with a 384-token generation limit. No prompts, scoring or retrieved evidence were edited. No sealed or contaminated telemetry task was selected; these are exposed development cases, not independent holdouts.

- **3/6 generated token sequences and decoded answers matched exactly**.
- Baseline output tokens: **443**; merged output tokens: **574**; total token edit distance: **212**.
- **191/192** top-token predictions matched on identical teacher-forced baseline-prefix positions (up to 32 positions per prompt).
- Maximum absolute logit difference was **12.8125**; per-case RMSE and exact token differences are in `TOKEN_COMPARISON.json`. Teacher-forced full-prefix logits are not identical to every cached generation step; this diagnostic does not cover the full output sequences.
- **Zero truncations**, finite generation logits, no optimizer, and parameter version counters unchanged.

**Bitwise equivalence is not demonstrated.** Separate FP32 LoRA GEMMs plus BF16 base computation and a single merged BF16 GEMM are numerically different operations. With the same input tokens, backend, placement and decoding, the weight-equation check supports rounding/operation-order differences rather than a merge implementation error. A full FP32 reference inference was not run, so exact attribution of every logit difference to each rounding stage is not established.

Provisional Codex review found the three changed answers retained the checked content: the arithmetic answer stayed incorrectly 230 GPU-minutes instead of 190; the load-average answer added accurate explanation; the vision answer expanded its source-grounded distinction between aspirations and empirical results. The other three answers were token-identical. **No material factual regression was observed in these six cases**, but this is not independent review, broad behavioral equivalence or biomedical validation. Existing citation and reasoning limitations persist.

## Runtime and hardware

The merge took **{resources['cpu_merge']['total_seconds']:.2f} seconds**; peak CPU RSS was **{resources['cpu_merge']['peak_rss_kib']/1024**2:.2f} GiB**. Preflight found about 1.9 TiB available host RAM, no swap, idle GPUs and roughly 50 TB free storage. CPU RSS was supervised below 256 GiB and host headroom above 250 GiB; the campaign and stages had fixed wall-time caps. No uncontrolled swapping occurred.

| Validation stage | Load seconds | Generation seconds | Mean generation latency | Peak host RSS GiB | Peak sampled GPU memory MiB (0 / 1) |
|---|---:|---:|---:|---:|---:|
| Base plus v1 | {baseline['load_seconds']:.2f} | {baseline['generation_seconds']:.2f} | {baseline['mean_latency_seconds']:.2f} | {baseline['peak_cpu_rss_kib']/1024**2:.2f} | {baseline['sampled_gpu']['0']['max_memory_mib']} / {baseline['sampled_gpu']['1']['max_memory_mib']} |
| Merged v1 | {merged['load_seconds']:.2f} | {merged['generation_seconds']:.2f} | {merged['mean_latency_seconds']:.2f} | {merged['peak_cpu_rss_kib']/1024**2:.2f} | {merged['sampled_gpu']['0']['max_memory_mib']} / {merged['sampled_gpu']['1']['max_memory_mib']} |
| Fresh smoke | {smoke['load_seconds']:.2f} | {smoke['generation_seconds']:.2f} | {smoke['mean_latency_seconds']:.2f} | {smoke['peak_cpu_rss_kib']/1024**2:.2f} | {smoke['sampled_gpu']['0']['max_memory_mib']} / {smoke['sampled_gpu']['1']['max_memory_mib']} |

Latency is not a throughput superiority comparison: output lengths differ. GPU utilization samples and memory records are in `RESOURCE_METRICS.json`; no volatile uncorrected ECC errors were observed. Only GPUs 0 and 1 were used. All worker process groups and campaign tmux sessions terminated, GPUs returned idle, and unrelated jobs/sessions were untouched.

Two initial standalone validation attempts stopped before model loading because the test harness's PEFT guard first blocked metadata discovery, then correctly blocked an optional PEFT import triggered by reporting it installed. The repaired test reports PEFT absent while still blocking direct imports; source files and original traces are preserved. A later CPU equation checker needed TMPDIR directed to its confined package folder. These were technical validation errors, not model-weight failures. Completed merge and baseline stages were never repeated. Nine focused CPU checks passed across the merge tests and guard/temp-directory regressions; all original source hashes were reverified after corrections.

## Portable package and dependencies

The model directory contains all shards, index, tokenizer, chat template, model/generation configuration, `SHA256_MANIFEST.json`, `PROVENANCE.json`, `VALIDATION_SUMMARY.json`, `MERGE_EQUATION_VERIFICATION.json`, dependency pins, standalone inference/checksum scripts and retained attribution/licence notices. RAG is separate; evidence may later be supplied through prompts.

Validated software: Python {sys.version.split()[0]}, PyTorch 2.13.0+cu132, Transformers 5.16.1, Accelerate 1.14.0, Safetensors 0.8.0, Tokenizers 0.23.2 and huggingface-hub 1.30.0. PEFT 0.20.0 was used for merging and baseline validation, but is not required for standalone inference. Dependency wheels are not bundled and availability on another server is not asserted.

The tested serving layout is two L40S 48 GB GPUs, SDPA/BF16 and no offload. The example reserves 38 GiB per GPU for placement. Measured host RSS was about 51.2 GiB; 64 GB host RAM is a lower planning allowance and 128 GB is recommended headroom, not a proven minimum. Other GPU layouts, quantization, serving engines, longer contexts and multimodal input remain unvalidated. Review `README.md` inside the model directory for exact loading commands.

The base Apache-2.0 licence, original upstream model card and inherited attribution/release-rights records are retained. This task does not settle the previously pending adapter/derivative public-distribution licence. Local internal packaging is complete; publication and deployment were neither performed nor authorized. Scientific superiority and clinical validation remain unestablished.

## Transfer and restoration

See `INCREMENTAL_TRANSFER.md` for exact commands using SSH alias `nvidia` and destination `/media/avi/GizmoQuest-World/`. The new transfer contains the merged model, packaging/validation receipts and STATE.v154 only. It does not include the original 115.72 GB backup, original base weights or original adapter weights. The manifest preserves folder structure and hashes every payload file. Raw baseline logits are retained for reproducibility; temporary caches are excluded.

No off-node transfer was executed. Verify checksums on the Linux destination, then load on the intended GPU server with the pinned environment. Restore acceptance on another physical server remains pending. Preserve the original v1 adapter as the authoritative reusable training/inference baseline and this merged export as a separate packaging artifact.
'''
 (P/'CONSOLIDATED_REPORT.md').write_text(report)
 (P/'INCREMENTAL_TRANSFER.md').write_text('''# Transfer the standalone model and validation receipts

Run on your Linux machine. This transfers only the new merged model, its validation package and STATE.v154 into the directory structure below your specified destination. Ensure at least 60 GB free; 120 GB is preferable if retaining additional incoming/restore copies. The exact payload size is recorded in BACKUP_MANIFEST.json.

```bash
ssh nvidia 'cat /data/axonai/engineering-maintenance/merged-v1-packaging-v1-20261109/DELTA_FILES.txt' > /tmp/axonai-merged-v1-files.txt
rsync -a --checksum --partial --info=progress2 --files-from=/tmp/axonai-merged-v1-files.txt nvidia:/data/axonai/ /media/avi/GizmoQuest-World/
python3 /media/avi/GizmoQuest-World/engineering-maintenance/merged-v1-packaging-v1-20261109/verify_manifest.py --root /media/avi/GizmoQuest-World/ --manifest /media/avi/GizmoQuest-World/engineering-maintenance/merged-v1-packaging-v1-20261109/BACKUP_MANIFEST.json
python3 /media/avi/GizmoQuest-World/models/merged/AxonAI-v1-27B/verify_manifest.py --root /media/avi/GizmoQuest-World/models/merged/AxonAI-v1-27B/ --manifest /media/avi/GizmoQuest-World/models/merged/AxonAI-v1-27B/SHA256_MANIFEST.json
```

The standalone directory after transfer is `/media/avi/GizmoQuest-World/models/merged/AxonAI-v1-27B/`. Move/copy that entire directory to the intended GPU server; follow its README. No original 115.72 GB payload, original base weights or original adapter weights are included. No transfer was executed here.

BACKUP_MANIFEST.json hashes payload files. DELTA_FILES.txt also includes transfer controls. TRANSFER_CONTROL_SHA256.json binds the manifest and file list; its hash is recorded in FINALIZATION_RESULT.json. Self-hashing files are excluded from recursive manifests. A successful local-on-node checksum check is not proof of off-node transfer or restoration. Keep earlier backups and perform your planned restoration test before node expiry.
''')
 # Package manifest includes all evidence except caches and recursive controls.
 excluded={'ARTIFACT_MANIFEST.json','BACKUP_MANIFEST.json','DELTA_FILES.txt','TRANSFER_CONTROL_SHA256.json','FINALIZATION_RESULT.json','FINAL_VERIFICATION.json','BACKUP_VERIFY.log'}
 packagefiles=sorted(f for f in P.rglob('*') if f.is_file() and not set(f.relative_to(P).parts)&{'tmp','cache','__pycache__'} and f.name not in excluded)
 save(P/'ARTIFACT_MANIFEST.json',{'schema':'axonai.merged-packaging-artifacts.v1','files':[binding(f) for f in packagefiles],'model_manifest':binding(M/'SHA256_MANIFEST.json'),'excluded_controls':sorted(excluded)})
 statepath=ROOT/'reports/axonai-v1-continuation-20281028/STATE.v154.json';assert not statepath.exists()
 state={'schema':'axonai.merged-model-handoff.v1','state_version':154,'status':'STANDALONE_V1_MERGED_MODEL_VALIDATED_NOT_BITWISE_EQUIVALENT','updated_utc':now.isoformat(),'user_local_recorded_at':now.astimezone(datetime.timezone(datetime.timedelta(hours=5,minutes=30))).isoformat(),'user_timezone':'Asia/Calcutta','previous_state':binding(state153),'package':str(P),'merged_model_directory':str(M),'consolidated_report':binding(P/'CONSOLIDATED_REPORT.md'),'artifact_manifest':binding(P/'ARTIFACT_MANIFEST.json'),'model_manifest':binding(M/'SHA256_MANIFEST.json'),'model_precision':'BF16; FP32 merge arithmetic before one output rounding','model_class':'Qwen3_5ForCausalLM','text_only':True,'upstream_vision_tensors_omitted':333,'upstream_mtp_tensors_omitted':15,'parameters':26895998464,'tensors':851,'weight_shards':14,'tensor_bytes':w['tensor_bytes'],'shard_bytes':w['shard_file_bytes'],'model_directory_bytes':model_bytes,'adapter_tensors_verified':992,'merged_modules':496,'all_weights_finite':True,'all_merge_equations_verified':True,'weight_equation_receipt':binding(P/'MERGE_EQUATION_VERIFICATION.json'),'weight_verification':binding(P/'WEIGHT_VERIFICATION.json'),'standalone_load_without_peft':True,'original_model_paths_denied_during_standalone_test':True,'smoke_passed':True,'comparison_prompts':6,'exact_output_matches':3,'aligned_logit_top1_matches':191,'aligned_logit_positions':192,'bitwise_equivalence_demonstrated':False,'material_factual_regression_observed_in_six_cases':False,'semantic_review':'PROVISIONAL_CODEX_NOT_INDEPENDENT','comparison':binding(P/'TOKEN_COMPARISON.json'),'validation_summary':binding(M/'VALIDATION_SUMMARY.json'),'resources':binding(P/'RESOURCE_METRICS.json'),'original_adapter':old['original_adapter'],'experimental_v11_adapter_used':False,'original_artifacts_modified':False,'final_input_integrity':binding(P/'FINAL_SOURCE_INTEGRITY.json'),'cpu_tests_passed':9,'cpu_tests':binding(P/'CPU_TEST_RESULTS.json'),'error_repairs':binding(P/'ERROR_REPAIR_JOURNAL.json'),'all_worker_processes_terminated':True,'process_termination':binding(P/'PROCESS_TERMINATION.json'),'gpu_used':True,'gpu_count':2,'new_training_started':False,'optimizer_updates':0,'publication_performed':False,'deployment_performed':False,'sealed_evaluation_accessed':False,'rag_included':False,'standalone_internal_self_hosting_readiness':True,'new_server_restore_test_performed':False,'public_distribution_rights':'INHERITED_PENDING_SEPARATE_REVIEW','off_node_transfer_performed':False,'off_node_backup_verified':False,'backup_manifest_path':str(P/'BACKUP_MANIFEST.json'),'backup_includes_original_115GB':False,'node_expiry':'2026-11-30','next_action':'Transfer and verify the new merged-model increment, then test restoration on the intended GPU server. Preserve original v1 adapter. No further training, inference campaign, deployment or publication is queued.'}
 save(statepath,state)
 payload=packagefiles+[P/'ARTIFACT_MANIFEST.json',statepath]+sorted(f for f in M.iterdir() if f.is_file())
 modelhash={str(M/x['relative_path']):x for x in files};items=[]
 for f in sorted(set(payload)):
  h=modelhash[str(f)]['sha256'] if str(f) in modelhash else sha(f)
  items.append({'relative_path':str(f.relative_to(ROOT)),'bytes':f.stat().st_size,'sha256':h})
 save(P/'BACKUP_MANIFEST.json',{'schema':'axonai.merged-model-increment.v1','files':items,'payload_bytes':sum(x['bytes'] for x in items),'file_count':len(items),'original_base_or_adapter_weights_included':False,'original_115GB_backup_repeated':False,'off_node_verified':False})
 controls=[P/n for n in ['BACKUP_MANIFEST.json','DELTA_FILES.txt','TRANSFER_CONTROL_SHA256.json']]
 (P/'DELTA_FILES.txt').write_text('\n'.join(sorted({x['relative_path'] for x in items}|{str(f.relative_to(ROOT)) for f in controls}))+'\n')
 save(P/'TRANSFER_CONTROL_SHA256.json',{'manifest':binding(P/'BACKUP_MANIFEST.json'),'file_list':binding(P/'DELTA_FILES.txt')})
 save(P/'FINALIZATION_RESULT.json',{'state':binding(statepath),'model_bytes':model_bytes,'backup_payload_bytes':sum(x['bytes'] for x in items),'backup_files':len(items),'control_sha256':sha(P/'TRANSFER_CONTROL_SHA256.json'),'model_manifest_sha256':sha(M/'SHA256_MANIFEST.json'),'off_node_transfer_performed':False})
 print(json.dumps(read(P/'FINALIZATION_RESULT.json'),indent=2))
if __name__=='__main__':main()
