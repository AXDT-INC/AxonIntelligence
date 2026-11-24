"""New independent replay verifier. No assembler import and no near-screen rerun."""
import collections
import difflib
import hashlib
import json
import platform
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from preflight import ROOT, HIST, PINS, PIN_SOURCE, OUTPUT_PINS, identity, digest

def equal_bytes(a,b):
    with a.open('rb') as fa,b.open('rb') as fb:
        while True:
            ba=fa.read(1024*1024);bb=fb.read(1024*1024)
            if ba!=bb:return False
            if not ba:return True

def normalize(s):
    return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())

def main():
    out=ROOT/'execution/output'
    pre=json.loads((ROOT/'logs/preflight.json').read_text())
    checks={};results={}
    exits=dict(x.split('=',1) for x in (ROOT/'logs/exit-status.txt').read_text().splitlines())
    checks['both_process_exit_codes_zero']=exits=={'assembler_exit':'0','tee_exit':'0'}
    for name,pin in OUTPUT_PINS.items():
        actual=identity(out/name)
        actual['expected_sha256']=pin
        actual['historical_file_sha256']=digest(HIST/name)
        actual['direct_byte_comparison']=equal_bytes(out/name,HIST/name)
        actual['pass']=actual['sha256']==pin==actual['historical_file_sha256'] and actual['direct_byte_comparison']
        results[name]=actual
    checks['five_historical_outputs_byte_identical']=all(x['pass'] for x in results.values())
    manifest=(out/'SHA256SUMS').read_text()
    pairs=[line.split(maxsplit=1) for line in manifest.splitlines()]
    checks['generated_manifest_correct']=len(pairs)==5 and {n:h for h,n in pairs}==OUTPUT_PINS
    checks['manifest_byte_identical']=equal_bytes(out/'SHA256SUMS',HIST/'SHA256SUMS')
    rows=[]; linecount=0
    with (out/'axonai-v1-20000-STAGING-NOT-FROZEN.jsonl').open('rb') as f:
        for line in f:
            linecount+=1
            assert line.endswith(b'\n') and len(line)<1024**2
            rows.append(json.loads(line))
    checks['record_count_and_bytes']=linecount==len(rows)==20000 and results['axonai-v1-20000-STAGING-NOT-FROZEN.jsonl']['bytes']==41846228
    canonical={'id','messages','domain','task_type','source','license','quality_score','provenance','verification'}
    structural=True
    for row in rows:
        v=row['verification'];msgs=row['messages']
        structural &= set(row)==canonical and [x['role'] for x in msgs]==['system','user','assistant']
        structural &= all(isinstance(x['content'],str) and x['content'].strip() for x in msgs)
        structural &= row['provenance'].get('split')=='train'
        structural &= v.get('fully_grounded') is True and v.get('unsupported_claims')==[]
        structural &= v.get('assigned_task_is_good_match') is True and v.get('question_quality',0)>=4 and v.get('answer_quality',0)>=4
    questions={r['id']:normalize(r['messages'][1]['content']) for r in rows}
    checks['independent_structural_check']=bool(structural) and len(questions)==len(set(questions.values()))==20000
    audit=json.loads((out/'staging-audit-report.json').read_text())
    checks['audit_identical']=audit==json.loads((HIST/'staging-audit-report.json').read_text())
    checks['domain_counts_independent']=dict(collections.Counter(r['domain'] for r in rows))==audit['domain_counts']
    checks['licence_counts_independent']=dict(collections.Counter(str(r['license']) for r in rows))==audit['licence_value_counts']
    checks['issuer_counts_independent']=dict(collections.Counter(r['source'] for r in rows))==audit['source_issuer_counts']
    checks['origin_counts']=audit['origin_counts']==dict(batch001=5800,pmc=7000,production=7160,historical_recoveries_28=28,duplicate_replacements_12=12)
    superseded=json.loads((out/'superseded-production-record-ids.json').read_text())
    production_ids=set()
    with PINS['production'][1].open() as f:
        for line in f:production_ids.add(json.loads(line)['id'])
    checks['supersession_list']=len(superseded)==len(set(superseded))==12 and superseded==sorted(superseded) and set(superseded)<=production_ids and not set(superseded)&set(questions)
    near=json.loads((out/'near-question-review.json').read_text())
    near_expected=json.loads((HIST/'near-question-review.json').read_text())
    checks['all_42_pairs_identical_including_order']=near==near_expected and len(near)==42
    checks['all_42_pairs_independently_verified']=all(
        x['question_1']==questions[x['id_1']][:1000] and x['question_2']==questions[x['id_2']][:1000]
        and x['ratio']==round(difflib.SequenceMatcher(None,questions[x['id_1']],questions[x['id_2']],autojunk=False).ratio(),6)
        and x['ratio']>=0.90 for x in near)
    method=audit['near_duplicate_method']
    counters={'pairs_considered':871180,'shards':64,'peak_shard_candidates':15233,'skipped_generic_shingles':115,'run':True,'is_exhaustive_semantic_paraphrase_check':False}
    checks['historical_near_screen_counters']=all(method[k]==v for k,v in counters.items())
    log=(ROOT/'logs/assembler.log').read_text()
    progress=[tuple(map(int,x)) for x in re.findall(r'NEAR_SCREEN progress=(\d+)/64 candidate_pairs=(\d+) flagged=(\d+)',log)]
    checks['complete_near_screen_progress']=progress==[(8,107813,7),(16,213083,10),(24,323991,16),(32,436084,23),(40,543635,32),(48,651693,35),(56,758406,37),(64,871180,42)]
    checks['nine_input_bindings']=json.loads((out/'input-sha256.json').read_text())=={k:v[0] for k,v in PINS.items()}
    preserved={}
    for key,(pin,path) in {'assembler':(PIN_SOURCE,ROOT/'inputs/axonai_stage_final_20000_v11.py'),**PINS}.items():
        post=identity(path)
        preserved[key]={'post':post,'sha256_matches_pin':post['sha256']==pin,
                       'identity_and_mtime_unchanged':post==pre['files'][key]}
    checks['all_source_and_input_bytes_preserved']=all(v['sha256_matches_pin'] and v['identity_and_mtime_unchanged'] for v in preserved.values())
    checks['staging_status_preserved']=audit['status']=='STAGING_COMPLETE_REVIEW_REQUIRED' and audit['ready_for_training'] is False
    result={'status':'PASS' if all(checks.values()) else 'FAIL','verified_at_utc':datetime.now(timezone.utc).isoformat(),
      'checks':checks,'exit_codes':exits,'outputs':results,'input_preservation':preserved,
      'record_count':linecount,'near_duplicate_pairs':len(near),'near_duplicate_method':method,
      'progress':progress,'start_utc':(ROOT/'logs/start-utc.txt').read_text().strip(),
      'end_utc':(ROOT/'logs/end-utc.txt').read_text().strip(),
      'source_sha256':PIN_SOURCE,'runtime':pre['runtime'],
      'git_reference_user_supplied_not_inspected':'530f0796a6c7dc22812245d327a0f3c2f3c80273',
      'note':'No private questions, answers, dataset rows or archive members included in this receipt.'}
    with (ROOT/'logs/replay-receipt.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'checks':checks,'outputs':{k:v['sha256'] for k,v in results.items()}}))
    assert all(checks.values()),'Output discrepancy: retained outputs and receipt; investigate failed checks without changing originals.'

if __name__=='__main__':main()
