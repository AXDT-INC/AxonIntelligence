"""Exact receipt binding for the project-owner-authorized main run."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MANIFEST='b0810d15a4aa2968b03690a63a6d38d0ba9f9a9162acd357d08f77dbe772c8a4'
DATA='641b43e32f3d69fa6fbf74754772e976777a26383c27de2692b90670c5dc5d07'
REVISION='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
CONFIG='43c6d60ee6805aa1b046b0828a51efffb3ea9d8543276deafa3a1141c06d3ef8'
WAIVER='548564741451d0cfa5c8a858a8319b2675b7ec71208422aa4f9b00bea7daa8fb'

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def bound(b):
    p=Path(b['path'])
    require(p.is_absolute() and p.is_file() and not p.is_symlink(),'unsafe/missing bound file')
    require(sha(p)==b['sha256'],'bound file hash mismatch: '+str(p))
    return json.loads(p.read_text()) if p.suffix=='.json' else None

def validate_rights(rights,amendment,reconciliation):
    require(rights.get('status')=='INTERNAL_USE_ELIGIBLE_BY_PROJECT_OWNER_AMENDMENT','rights decision not adopted')
    require(rights.get('manifest_sha256')==MANIFEST and rights.get('record_count')==10684,'rights scope mismatch')
    require(rights.get('main_sft_rights_gate_satisfied') is True,'rights gate not satisfied')
    require(rights.get('applicable_licence_obligations_preserved') is True and rights.get('new_concrete_restrictions_must_block') is True,'licence obligations missing')
    for key in ('independent_legal_clearance','third_party_permission_created','historical_holds_cleared','publication_authorized','external_mandate_established'):
        require(rights.get(key) is False,'invalid rights assertion: '+key)
    require(amendment.get('status')=='ADOPTED_PROSPECTIVE_INTERNAL_POLICY','owner amendment not adopted')
    require(amendment.get('external_independent_signoff_mandate_established') is False,'external mandate unresolved')
    require(reconciliation.get('status')=='PASS_EXISTING_EVIDENCE_RECONCILIATION' and reconciliation.get('records')==10684 and reconciliation.get('exact_id_and_pmcid_coverage') is True and reconciliation.get('concrete_exclusions_identified')==0,'cohort reconciliation failed')

def verify_approval(path,digest,root=ROOT):
    a=bound({'path':str(Path(path).resolve()),'sha256':digest})
    require(a.get('status')=='APPROVED_MAIN_SFT' and a.get('active') is True,'inactive or wrong approval')
    require(a.get('mode')=='MAIN_SFT' and a.get('dataset_sha256')==MANIFEST and a.get('training_jsonl_sha256')==DATA,'approval scope mismatch')
    require(a.get('runtime_limit_seconds')==86400 and a.get('optimizer_steps')==668 and a.get('warmup_steps')==21 and a.get('world_size')==8 and a.get('epochs')==2,'execution limits mismatch')
    require(a.get('model_revision')==REVISION,'model revision mismatch')
    require(Path(a['snapshot_manifest']['path']).resolve()==root/'SNAPSHOT.json','wrong snapshot')
    snapshot=bound(a['snapshot_manifest'])
    expected={'train.py','launcher.py','supervisor.py','guard.py','schedule.py','shared_engine.py','config.fsdp1.json','contract.json'}
    require(set(snapshot['files'])==expected,'snapshot source inventory mismatch')
    for name,h in snapshot['files'].items():bound({'path':str(root/name),'sha256':h})
    require(a.get('worker_sha256')==snapshot['files']['train.py'] and a.get('config_sha256')==CONFIG and snapshot['files']['config.fsdp1.json']==CONFIG,'worker/config approval mismatch')
    contract=json.loads((root/'contract.json').read_text())
    required={'declaration','amendment','rights','reconciliation','acceptance','freeze','waiver','manifest','dataset','extraction','evaluation_seal','model_integrity','investigation'}
    require(required.issubset(contract['bindings']),'missing contract bindings')
    values={key:bound(value) for key,value in contract['bindings'].items()}
    b=contract['bindings']
    previous=values['prior_startup_failure']
    require(previous['optimizer_steps']==0 and previous['gpu_cleanup_verified'] is True,'previous attempt not safely closed')
    require(a.get('prior_attempt_runtime_seconds')==previous['supervised_runtime_seconds'] and 0<=a['prior_attempt_runtime_seconds']<86400,'prior runtime budget mismatch')
    require(b['dataset']['sha256']==DATA and b['manifest']['sha256']==MANIFEST and b['waiver']['sha256']==WAIVER,'frozen artifact pin mismatch')
    validate_rights(values['rights'],values['amendment'],values['reconciliation'])
    require(values['rights']['amendment']==b['amendment'] and values['rights']['reconciliation']==b['reconciliation'] and values['rights']['prior_acceptance']==b['acceptance'],'decision evidence link mismatch')
    require(values['amendment']['authority_source']==b['declaration'] and values['amendment']['investigation']==b['investigation'],'owner authority link mismatch')
    require(a.get('rights_determination')==b['rights'] and a.get('owner_declaration')==b['declaration'],'approval authority binding mismatch')
    require(values['freeze'].get('status')=='DATASET_FROZEN' and values['freeze'].get('manifest_sha256')==MANIFEST,'freeze inactive')
    e=values['extraction']
    require(e.get('status')=='COMPLETE' and e.get('approved_manifest_sha256')==MANIFEST and e.get('extracted_jsonl_sha256')==DATA and e.get('record_count')==10684 and e.get('id_coverage_exact') is True,'extraction mismatch')
    require(values['waiver'].get('training_time_evaluation')=='disabled' and values['evaluation_seal'].get('status')=='EVALUATION_SEALED','evaluation policy mismatch')
    model=values['model_integrity']
    require(model['status']=='PASS_PINNED_LOCAL_BYTES_AND_SAVED_DOWNLOAD_ETAGS' and model['revision']==REVISION,'model verification incomplete')
    for entry in model['files']:
        stat=Path(entry['path']).stat()
        require((stat.st_size,stat.st_mtime_ns,stat.st_ino,stat.st_dev)==(entry['bytes'],entry['mtime_ns'],entry['inode'],entry['device']),'model changed since final full hash verification')
    cfg=json.loads((root/'config.fsdp1.json').read_text())
    from schedule import resolve_schedule
    require(resolve_schedule(10684)['total_steps']==668 and cfg['warmup_steps']==21 and cfg['num_train_epochs']==2,'schedule mismatch')
    return a,contract

def verify_worker(args):
    run=Path(args.run_dir).resolve()
    auth=json.loads((run/'LAUNCH_AUTHORIZATION.json').read_text())
    a,c=verify_approval(auth['approval'],auth['approval_sha256'])
    require(run==Path(a['output']).resolve(),'unauthorized output directory')
    require(Path(args.config).resolve()==ROOT/'config.fsdp1.json','worker config path mismatch')
    require(Path(args.train_file).resolve()==Path(c['bindings']['dataset']['path']),'worker dataset path mismatch')
    require(args.validation_file is None and args.evaluation_waiver and Path(args.evaluation_waiver).resolve()==Path(c['bindings']['waiver']['path']),'worker evaluation policy mismatch')
    require(not args.resume,'main launch cannot implicitly resume')
    require(auth['identity_sha256']==sha(run/'identity.json'),'identity hash mismatch')
    identity=json.loads((run/'identity.json').read_text())
    require(identity.get('approval_sha256')==auth['approval_sha256'] and identity.get('snapshot_sha256')==a['snapshot_manifest']['sha256'],'worker approval/snapshot identity mismatch')
    return a
