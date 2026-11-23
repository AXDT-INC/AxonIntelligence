"""Record identity/source lineage only; no scientific or rights reassessment."""
import json,hashlib,re,collections,os
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/data/axonai');HERE=Path(__file__).resolve().parent;ENG=ROOT/'engineering-maintenance'
GEN=ROOT/'datasets/axonai-v1/generated/20k-completion-production/run-v1'
PLAN=ROOT/'datasets/axonai-v1/plans/20k-completion'
PATHS={'staging_20000':GEN/'recovery/final-20000-staging-v11-20261009T051315Z/axonai-v1-20000-STAGING-NOT-FROZEN.jsonl','isolated_19257':ENG/'isolated-candidate-v1-20281029/candidate-19257.jsonl','trained_10684':ENG/'frozen-v1-training-extraction-v1-20281029/train-10684.jsonl','separate_axonos_600':ROOT/'datasets/sft/axonai-v1/generated/production/accepted.jsonl','completion_accepted_7172':GEN/'accepted-canonical.jsonl','frozen_slots_7200':PLAN/'final-routing-freeze/freeze-20261003T030006Z/final-generation-source-manifest-7200.jsonl'}
INPUTS={}
def digest(b):return hashlib.sha256(b).hexdigest()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bind(p):return {'path':str(p),'bytes':Path(p).stat().st_size,'sha256':sha(p)}
def track(p):INPUTS[str(p)]=bind(p);return p
def read(p):return json.loads(track(p).read_text())
def rows(p):
 track(p)
 with p.open('rb') as f:return [(json.loads(l),{'line':i,'raw_line_sha256':digest(l)}) for i,l in enumerate(f,1) if l.strip()]
def save(name,obj):
 with (HERE/name).open('x') as f:json.dump(obj,f,indent=2,sort_keys=True,ensure_ascii=False);f.write('\n')
def write_rows(name,rs):
 with (HERE/name).open('x') as f:
  for x in rs:f.write(json.dumps(x,sort_keys=True,ensure_ascii=False)+'\n')
def body(r):return '\n'.join(m['content'] for m in r.get('messages',[]) if m['role'] in ['user','assistant'])
def canon(x):return digest(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode())
def qhash(r):return digest(re.sub(r'\s+',' ',next((m['content'] for m in r.get('messages',[]) if m['role']=='user'),'')).strip().casefold().encode())
PATTERNS={'axonos':r'\baxonos\b','axondao':r'\baxon[\s_-]?dao\b','desci':r'\bdesci(?:ence)?\b|\bdecentrali[sz]ed science\b'}
def signals(r):return {k:bool(re.search(pattern,body(r),re.I)) for k,pattern in PATTERNS.items()}

def main():
 data={k:rows(p) for k,p in PATHS.items()};by={k:{r.get('id',r.get('slot_id')):r for r,_ in v} for k,v in data.items()}
 for k,v in data.items():assert len(by[k])==len(v),k
 a,b,c=by['staging_20000'],by['isolated_19257'],by['trained_10684'];sa,sb,sc=map(set,[a,b,c]);assert sc<=sb<=sa
 exclusion=read(ENG/'isolated-candidate-v1-20281029/exclusion-manifest.json');assert sa-sb==set(exclusion['excluded_ids'])
 manifest=read(ENG/'v1-revised-scope-package-v1-20281029/V1_PROPOSED_MANIFEST.json');assert sc==set(manifest['record_ids'])
 assert [r['id'] for r,_ in data['trained_10684']]==manifest['record_ids']
 hold=read(ENG/'human-review-dossier-v1-20281029/RIGHTS_RESOLUTION_REGISTER.json');holds={}
 for group in hold['source_groups']:
  for x in group['affected_records']:holds[x['record_id']]={'source_document':group['source_document'],'recorded_decision':x.get('decision'),'recorded_hold':x.get('hold'),'required_decision':group.get('required_decision')}
 stats={}
 for key,records in data.items():
  rr=[r for r,_ in records];systems=collections.Counter(m['content'] for r in rr for m in r.get('messages',[]) if m['role']=='system')
  stats[key]={'binding':INPUTS[str(PATHS[key])],'records':len(rr),'source_counts':dict(collections.Counter(r.get('source','MISSING') for r in rr)),'domain_counts':dict(collections.Counter(r.get('domain','MISSING') for r in rr)),'task_counts':dict(collections.Counter(r.get('task_type','MISSING') for r in rr)),'system_prompts':[{'sha256':digest(text.encode()),'records':n,'text':text} for text,n in systems.items()],'user_target_lexical_records':{tag:sum(signals(r)[tag] for r in rr) for tag in PATTERNS},'lexical_counts_are_not_semantic_coverage':True}
 classifications=[]
 for r,pos in data['staging_20000']:
  rid=r['id'];state='TRAINED' if rid in sc else 'EXCLUDED_IN_743_QUARANTINE' if rid not in sb else 'DEFERRED_IN_PMC_ONLY_SCOPE'
  classifications.append({'id':rid,**pos,'source':r['source'],'original_domain':r['domain'],'task_type':r['task_type'],'status':state,'present_in_isolated_19257':rid in sb,'present_in_trained_10684':rid in sc,'source_unit_id':r.get('provenance',{}).get('source_unit_id'),'frozen_slot_id':r.get('provenance',{}).get('frozen_slot_id'),'commit':r.get('provenance',{}).get('commit'),'source_file':r.get('provenance',{}).get('source_file'),'body_lexical_signals':signals(r),'existing_hold_evidence':holds.get(rid),'message_sha256':canon(r['messages']),'candidate_message_unchanged':canon(b[rid]['messages'])==canon(r['messages']) if rid in sb else None,'trained_message_unchanged':canon(c[rid]['messages'])==canon(r['messages']) if rid in sc else None})
 write_rows('RECORD_LINEAGE.jsonl',classifications)
 transition=[]
 for source in sorted({r['source'] for r in a.values()}):
  sourceids={i for i,r in a.items() if r['source']==source}
  transition.append({'source':source,'staging_20000':len(sourceids),'quarantined_743':len(sourceids-sb),'isolated_19257':len(sourceids&sb),'deferred_8573':len(sourceids&(sb-sc)),'trained_10684':len(sourceids&sc)})
 changed=[]
 for rid in sc:
  if a[rid]!=c[rid]:changed.append({'id':rid,'changed_top_level_keys':[k for k in set(a[rid])|set(c[rid]) if a[rid].get(k)!=c[rid].get(k)],'original_domain':a[rid]['domain'],'trained_domain':c[rid]['domain'],'messages_equal':a[rid]['messages']==c[rid]['messages']})
 slots=by['frozen_slots_7200'];finalslots={r.get('provenance',{}).get('frozen_slot_id'):r for r in a.values() if r.get('provenance',{}).get('frozen_slot_id')}
 changes=[]
 for sid,slot in slots.items():
  if slot['source']!='AXDT-INC/AxonOS':continue
  final=finalslots[sid]
  if final['source']!=slot['source']:changes.append({'slot_id':sid,'original_source':slot['source'],'original_source_file':slot.get('source_file'),'final_id':final['id'],'final_source':final['source'],'recovery_provenance':final['provenance']})
 write_rows('AXONOS_SLOT_REPLACEMENTS.jsonl',changes)
 separate=by['separate_axonos_600'];separate_q={qhash(r) for r in separate.values()};staging_q={qhash(r) for r in a.values()};trained_q={qhash(r) for r in c.values()}
 separate_details={'records':len(separate),'id_overlap_staging':len(set(separate)&sa),'id_overlap_training':len(set(separate)&sc),'normalized_question_overlap_staging':len(separate_q&staging_q),'normalized_question_overlap_training':len(separate_q&trained_q),'repository_commits':dict(collections.Counter(r.get('commit','MISSING') for r in separate.values())),'human_reviewed_true':sum(r.get('generation',{}).get('human_reviewed') is True for r in separate.values()),'not_used_in_completed_main_sft':True,'no_claim_about_all_historical_smoke_runs':True,'semantic_overlap_not_assessed':True}
 savedholds=collections.Counter(x['existing_hold_evidence']['source_document'] if x['existing_hold_evidence'] else 'ADDITIONAL_CURRENT_PYMOL_UNCERTAINTY_PER_QUARANTINE_PLAN' for x in classifications if x['source']=='AXDT-INC/AxonOS' and x['status']=='EXCLUDED_IN_743_QUARANTINE')
 save('LINEAGE_COUNTS.json',{'datasets':stats,'source_transitions':transition,'stage_counts':[20000,19257,10684],'quarantined':len(sa-sb),'deferred':len(sb-sc),'total_not_trained':len(sa-sc),'axonos_slot_source_changes':len(changes),'metadata_changes_baseline_to_trained':changed,'separate_axonos':separate_details,'axonos_quarantine_existing_reasons':dict(savedholds),'causality':'Exact ID membership proves exclusion from this SFT, not that the base lacks a capability or that the adapter cannot answer it.'})
 write_rows('LEXICAL_EVIDENCE_INDEX.jsonl',[{'dataset':k,'id':r.get('id'),'signals':signals(r),'question':next((m['content'] for m in r.get('messages',[]) if m['role']=='user'),'')[:450]} for k,rs in data.items() if k not in ['frozen_slots_7200','completion_accepted_7172'] for r,_ in rs if signals(r)['axondao'] or signals(r)['desci']])
 # Bounded filesystem discovery. No sealed/evaluation/credential/model/source-code payloads.
 roots=['datasets','remediation-runs','verification-runs','closure-runs','decision-runs','rehabilitation-runs','training','engineering-maintenance']
 found=[];examined=0
 for top in roots:
  for directory,subdirs,names in os.walk(ROOT/top):
   subdirs[:]=[x for x in subdirs if not x.startswith('.') and x not in ['fixtures','__pycache__','critical-staging','release-repository'] and not any(t in x.lower() for t in ['eval','benchmark','smoke','test','source-alignment','source-context-grounding','release-window-preparation','scope-lineage-investigation'])]
   for name in names:
    p=Path(directory)/name
    if p.suffix!='.jsonl' or p.is_symlink():continue
    examined+=1
    with p.open('rb') as f:first=f.readline(2_000_001)
    if len(first)>2_000_000:continue
    try:obj=json.loads(first)
    except (ValueError,UnicodeDecodeError):continue
    if isinstance(obj,dict) and isinstance(obj.get('messages'),list):found.append({'path':str(p),'bytes':p.stat().st_size,'first_record_id':obj.get('id'),'first_record_source':obj.get('source'),'first_record_domain':obj.get('domain'),'interpretation':'Candidate/derived message dataset; not automatically unique or training-approved.'})
 save('DATASET_DISCOVERY.json',{'roots':[str(ROOT/x) for x in roots],'jsonl_files_header_examined':examined,'message_dataset_files':found,'raw_descience_files':len([p for p in (ROOT/'datasets/axonai-v1/raw/descience').rglob('*') if p.is_file()]),'excluded':'Sealed/evaluation/benchmark/smoke/test directories, credentials, models, backups, source-code vendors and this investigation. JSONL top-level messages discovery only; archives/SQLite not searched for hypothetical alternate training sets.','absence_claim_limited_to_searched_scope':True})
 # Pin existing decision evidence, without revisiting decisions.
 evid=[ROOT/'AXONAI_V1_CODEX_HANDOFF_20261009.md',ROOT/'datasets/sft/axonai-v1/schema.json',PLAN/'plan-20261003T000522Z/completion-plan.json',PLAN/'source-feasibility-v2.json',PLAN/'feasibility-rebalance-v31-20261003T005829Z.json',PLAN/'frozen-7200/freeze-v11-20261003T013409Z/hard-audit-report.json',ENG/'v1-revised-scope-package-v1-20281029/V1_SCOPE_RECEIPT.json',ENG/'v1-freeze-activation-v1-20281029/PROJECT_AUTHORITY_APPROVAL.md',ENG/'v1-freeze-activation-v1-20281029/DATASET_FREEZE_RECEIPT.json',ENG/'project-owner-governance-v1-20261029/OWNER_DECLARATION.v1.json',ENG/'project-owner-governance-v1-20261029/GOVERNANCE_AMENDMENT.v1.json',ENG/'frozen-v1-training-extraction-v1-20281029/EXTRACTION_RECEIPT.json',ROOT/'datasets/sft/axonai-v1/axonos_corpus/train_sources.jsonl',ROOT/'datasets/sft/axonai-v2-main/axonos_corpus/train_sources.jsonl',ROOT/'datasets/sft/axonai-v2-main/architecture_analysis/summary.json']
 evid.extend(ROOT/'src'/n for n in ['axonai_plan_20k_completion-20261003T000522Z.py','axonai_source_feasibility_v2.py','axonai_freeze_7200_source_task_v1-20261003T012843Z.py','axonai_freeze_7200_source_task_v11-20261003T013409Z.py','axonai_production_7200_v1.py','generate_axonos_production_v2.py','build_axonos_main_corpus.py'])
 for n in [37,38,46,48,49,55,56,57,61,62,63,64,138,139,140,141,142,143,144,145]:
  p=ROOT/f'reports/axonai-v1-continuation-20281028/STATE.v{n}.json'
  if p.exists():evid.append(p)
 for p in evid:track(p)
 for p in INPUTS.values():assert bind(p['path'])==p,p['path']
 save('INPUT_MANIFEST.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'files':list(INPUTS.values()),'originals_modified':False,'all_analyzed_input_hashes_reverified':True})
 print(json.dumps({'source_transitions':transition,'separate_axonos':separate_details,'changed_metadata_count':len(changed),'message_changes':sum(not x['messages_equal'] for x in changed),'lexical':{k:v['user_target_lexical_records'] for k,v in stats.items()},'discovered_message_datasets':len(found),'axonos_quarantine_existing_reasons':dict(savedholds)},indent=2))

if __name__=='__main__':main()
