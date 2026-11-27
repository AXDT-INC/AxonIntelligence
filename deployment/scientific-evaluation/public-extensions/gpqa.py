"""Append-only, private GPQA extension to an already-running FP16 comparison.

This module lives outside the active worker package so its immutable code freeze
is not changed. The extension queues behind the original run on the same GPU pair.
"""
import argparse
from collections import Counter
import csv
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sqlite3
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
WORKER=HERE.parent/'fp16-worker'
sys.path.insert(0,str(WORKER))
import public_suite as suite
import benchmark_run as runner

REVISION='83022cefff930aea54f654c0b282e74b9eeda5c6'


def save(path,data):
    pending=path.with_suffix(path.suffix+'.tmp')
    with pending.open('w') as f:
        json.dump(data,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
    pending.replace(path)


def permutation(item_id):
    return sorted(range(4),key=lambda i:(hashlib.sha256(f'42/{item_id}/{i}'.encode()).digest(),i))


def project(row):
    ident='gpqa-'+row['Record ID'].strip()
    assert 0<len(ident)<=128 and ident!='gpqa-'
    options=[row['Correct Answer'],*[row[f'Incorrect Answer {i}'] for i in range(1,4)]]
    assert all(isinstance(x,str) and x.strip() for x in options) and row['Question'].strip()
    order=permutation(ident)
    return dict(id=ident,suite='gpqa',family='gpqa-record:'+row['Record ID'].strip(),
                gold='abcd'[order.index(0)],choice_order=order,question=row['Question'],
                subject=row['High-level domain'],prompt=suite.render('gpqa',row['Question'],options=[options[i] for i in order]))



def write_roster(path,rows):
    # Escape Unicode line/paragraph separators: the frozen worker's JSONL readers
    # use str.splitlines(), which recognizes these inside otherwise valid JSON.
    with path.open('x') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=True)+'\n')


def parent_progress(parent):
    db=sqlite3.connect('file:'+str(parent/'attempts.sqlite3')+'?mode=ro',uri=True)
    columns='item_id,arm,block,request_sha256,claim_hmac,state,claimed_at,finished_at,outcome'
    completed={ident+'|'+arm:hashlib.sha256(suite.canonical([ident,arm,*rest])).hexdigest()
               for ident,arm,*rest in db.execute('SELECT '+columns+" FROM attempts WHERE state='completed'")}
    return {'roster_sha256':suite.sha(parent/'roster.jsonl'),'completed_row_sha256':completed,
            'states':dict(db.execute('SELECT state,count(*) FROM attempts GROUP BY state'))}


def verify_parent_progress(parent,before):
    after=parent_progress(parent)
    assert before['roster_sha256']==after['roster_sha256']
    assert all(after['completed_row_sha256'].get(k)==v for k,v in before['completed_row_sha256'].items())
    return after


def prepare(parent,root):
    from transformers import AutoTokenizer
    assert not (root/'roster.jsonl').exists(), 'Extension already frozen'
    config=json.loads((parent/'run-config.json').read_bytes())
    before=parent_progress(parent);save(root/'parent-progress-before.json',before)
    assert json.loads((parent/'qualification.json').read_bytes())['code_hashes']==runner.code_hashes()
    with (root/'sources/gpqa/gpqa_diamond.csv').open(newline='') as f:
        data=list(csv.DictReader(f))
    assert len(data)==198 and len({r['Record ID'].strip() for r in data})==198
    rows=[project(row) for row in data]
    assert len({r['id'] for r in rows})==198
    tokenizers=[AutoTokenizer.from_pretrained(config['models'][arm],local_files_only=True,trust_remote_code=False) for arm in ('qwen','axonai')]
    for row in rows:
        messages=[{'role':'system','content':suite.SYSTEM},{'role':'user','content':row['prompt']}]
        tokens=[t.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False) for t in tokenizers]
        assert tokens[0]==tokens[1], 'Cross-arm tokenization differs'
        row.update(input_tokens=len(tokens[0]),input_ids_sha256=suite.ids_digest(tokens[0]))
        row['eligible']=len(tokens[0])<=2048 and '<|im_start|>' not in row['prompt'] and '<|im_end|>' not in row['prompt']
        row['exclusion']=None if row['eligible'] else 'context_overflow' if len(tokens[0])>2048 else 'reserved_delimiter'
    random.Random(42).shuffle(rows)
    for i,row in enumerate(rows):row.update(block=i//99,order=i)
    write_roster(root/'roster.jsonl',rows)
    freeze={'planned':{'gpqa':198},'eligible':{'gpqa':sum(r['eligible'] for r in rows)},
            'excluded':{'gpqa':sum(not r['eligible'] for r in rows)},'arms':['qwen','axonai'],
            'roster_sha256':suite.sha(root/'roster.jsonl'),'ordered_ids_sha256':hashlib.sha256(suite.canonical([r['id'] for r in rows])).hexdigest(),
            'renderer_sha256':suite.sha(WORKER/'public_suite.py'),'extension_code_sha256':suite.sha(__file__),
            'parent_roster_sha256':before['roster_sha256'],'source_revision':REVISION,'seed':42,
            'source_files':{str(p.relative_to(root)):suite.sha(p) for p in sorted((root/'sources').rglob('*')) if p.is_file()},
            'input_limit':2048,'output_limit':256,'block_arm_order':[['qwen','axonai'],['axonai','qwen']],
            'phase':'additive_followup_on_same_gpu_pair','individual_examples_and_raw_outputs':'private_local_only'}
    save(root/'roster-freeze.json',freeze)
    for name in ('run-config.json','qualification.json'):shutil.copy2(parent/name,root/name)
    sys.path.insert(0,str(parent/'screen-tools'))
    import screen_public
    screen_public.main(root)
    after=verify_parent_progress(parent,before)
    save(root/'parent-progress-preservation.json',{'verified_at':runner.now(),'previous_completed_retained':len(before['completed_row_sha256']),'current_states':after['states'],'original_roster_unchanged':True,'original_worker_code_unchanged':runner.code_hashes()==json.loads((parent/'qualification.json').read_bytes())['code_hashes']})
    save(root/'queue-status.json',{'state':'prepared','planned':198,'eligible':freeze['eligible']['gpqa'],'scheduled_answers':2*freeze['eligible']['gpqa'],'parent_roster_sha256':before['roster_sha256']})
    print(json.dumps({'planned':198,'eligible':freeze['eligible']['gpqa'],'max_input_tokens':max(r['input_tokens'] for r in rows),'existing_completed_preserved':len(before['completed_row_sha256'])}),flush=True)


def analyze_extension(root):
    previous=suite.SUITES
    try:
        suite.SUITES={'gpqa':198}
        result=suite.analyze(root)
    finally:suite.SUITES=previous
    result.update(primary=None,gpqa='included',interpretation='Descriptive GPQA Diamond extension; the primary PubMedQA contrast belongs to the original phase.')
    save(root/'results.json',result)
    return result


def combine(parent,root):
    original=json.loads((parent/'results.json').read_bytes());extension=json.loads((root/'results.json').read_bytes())
    result={'primary':'pubmedqa','suites':{**original['suites'],**extension['suites']},
            'phases':{'original':{'results_sha256':suite.sha(parent/'results.json'),'roster_sha256':suite.sha(parent/'roster.jsonl')},
                      'gpqa_extension':{'results_sha256':suite.sha(root/'results.json'),'roster_sha256':suite.sha(root/'roster.jsonl')}},
            'interpretation':'GPQA was added prospectively during the original collection and collected later on the same pair; existing answers were retained unchanged.',
            'overlap_screened_sensitivity':'unavailable: incomplete corpus and family-crosswalk coverage',
            'individual_examples_and_raw_records':'private_local_only'}
    save(parent/'combined-results.json',result)
    completions={name:json.loads(path.read_bytes()) for name,path in [('original',parent/'completion.json'),('gpqa',root/'completion.json')]}
    save(parent/'combined-completion.json',{'completed_at':runner.now(),'planned_questions':6200,'scheduled_answers':sum(s['eligible'] for s in result['suites'].values())*2,'phases':completions,'combined_results_sha256':suite.sha(parent/'combined-results.json')})


def wait_and_collect(parent,root):
    freeze=json.loads((root/'roster-freeze.json').read_bytes())
    assert freeze['extension_code_sha256']==suite.sha(__file__)
    assert freeze['parent_roster_sha256']==suite.sha(parent/'roster.jsonl')
    assert freeze['roster_sha256']==suite.sha(root/'roster.jsonl')
    # Parent is not interrupted and its ledger/roster/qualification are never rewritten.
    save(root/'queue-status.json',{'state':'waiting_for_original_collection','updated_at':runner.now(),'pid':os.getpid(),'same_gpu_pair':True})
    while not (parent/'completion.json').exists():
        time.sleep(30)
    config=json.loads((root/'run-config.json').read_bytes());locks=[]
    for uuid in sorted(config['gpu_uuids']):
        lock=open('/tmp/axonai-benchmark-'+uuid+'.lock','a');fcntl.flock(lock,fcntl.LOCK_EX);locks.append(lock)
    assert json.loads((parent/'completion.json').read_bytes())['all_slots_attempted']
    assert runner.code_hashes()==json.loads((root/'qualification.json').read_bytes())['code_hashes']
    before=json.loads((root/'parent-progress-before.json').read_bytes());verify_parent_progress(parent,before)
    save(root/'queue-status.json',{'state':'collecting','updated_at':runner.now(),'pid':os.getpid()})
    runner.analyze=analyze_extension
    runner.collect(root,config)
    combine(parent,root)
    save(root/'queue-status.json',{'state':'complete','updated_at':runner.now(),'results_sha256':suite.sha(root/'results.json')})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','queue','analyze']);parser.add_argument('parent',type=Path);args=parser.parse_args()
    os.umask(0o077);parent=args.parent.resolve();root=parent/'gpqa-extension'
    if args.command=='prepare':prepare(parent,root)
    elif args.command=='queue':
        try:wait_and_collect(parent,root)
        except BaseException as error:
            save(root/'queue-status.json',{'state':'failed','updated_at':runner.now(),'error_type':type(error).__name__})
            raise
    else:analyze_extension(root)


if __name__=='__main__':main()
