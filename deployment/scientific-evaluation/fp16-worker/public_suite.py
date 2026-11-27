"""Private public-suite roster, strict answer parsing, and paired local analysis.

No model calls here. Source data, keys, and responses stay in the private run root.
"""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import random
import re

SYSTEM = 'You are a scientific research assistant. Follow the requested answer format.'
SUITES = {'pubmedqa': 500, 'medmcqa': 4183, 'gsm8k': 1319}
NUMBER = re.compile(r'[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?\Z')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def ids_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def number(text):
    if not NUMBER.fullmatch(text):
        return None
    try:
        value = Decimal(text.replace(',', ''))
        return format(value.normalize(), 'f') if value else '0'
    except InvalidOperation:
        return None


def parse_answer(text, suite):
    """Exactly one final line, no trailing prose, no conflicting FINAL markers."""
    lines = text.strip().splitlines()
    candidates = [i for i, line in enumerate(lines) if re.match(r'^\s*FINAL\b', line, re.I)]
    if len(candidates) != 1 or candidates[0] != len(lines)-1:
        return None
    match = re.fullmatch(r'\s*FINAL\s+(.+?)\s*', lines[-1], re.I)
    if not match:
        return None
    value = match[1].strip()
    if suite == 'gsm8k':
        return number(value)
    allowed = {'yes', 'no', 'maybe'} if suite == 'pubmedqa' else {'a', 'b', 'c', 'd'}
    return value.casefold() if value.casefold() in allowed else None


def render(suite, question, context=None, options=None):
    parts = []
    if context is not None:
        parts.append('Context:\n' + '\n'.join(context))
    parts.append('Question:\n' + question)
    if options is not None:
        assert len(options) == 4
        parts.append('\n'.join(f'{chr(65+i)}. {option}' for i, option in enumerate(options)))
    parts.append('Return only the final line.' if suite in ('pubmedqa', 'medmcqa')
                 else 'You may include brief working before the final line.')
    label = 'yes, no, or maybe' if suite == 'pubmedqa' else 'a number' if suite == 'gsm8k' else 'A, B, C, or D'
    parts.append('End with exactly one line: FINAL ' + label + '.')
    return '\n\n'.join(parts).replace('\r\n', '\n').replace('\r', '\n')


def read_rows(root):
    import pyarrow.parquet as pq
    sources = root/'sources'
    data = json.loads((sources/'pubmedqa/ori_pqal.json').read_bytes())
    key = json.loads((sources/'pubmedqa/test_ground_truth.json').read_bytes())
    assert len(key) == 500
    rows = []
    for pmid in sorted(key):
        row = data[pmid]
        assert key[pmid] == row['final_decision'] and key[pmid] in ('yes','no','maybe')
        rows.append(dict(id='pubmedqa-'+pmid, suite='pubmedqa', family='pmid:'+pmid,
                         gold=key[pmid], question=row['QUESTION'], contexts=row['CONTEXTS'],
                         prompt=render('pubmedqa', row['QUESTION'], row['CONTEXTS'])))
    med = pq.read_table(sources/'medmcqa/validation-00000-of-00001.parquet').to_pylist()
    assert len(med) == 4183 and {r['cop'] for r in med} == {0,1,2,3}
    for row in med:
        options = [row['op'+c] for c in 'abcd']
        rows.append(dict(id='medmcqa-'+row['id'], suite='medmcqa', family='medmcqa:'+row['id'],
                         gold='abcd'[row['cop']], subject=row['subject_name'], question=row['question'],
                         prompt=render('medmcqa', row['question'], options=options)))
    gsm = pq.read_table(sources/'gsm8k/test-00000-of-00001.parquet').to_pylist()
    assert len(gsm) == 1319
    for i, row in enumerate(gsm):
        gold = number(row['answer'].rsplit('####', 1)[1].strip())
        assert gold is not None
        rows.append(dict(id=f'gsm8k-{i:04}', suite='gsm8k', family=f'gsm8k:{i}', gold=gold,
                         question=row['question'], prompt=render('gsm8k', row['question'])))
    assert len({r['id'] for r in rows}) == len(rows) == sum(SUITES.values())
    return rows


def prepare(root, axonai, qwen):
    from transformers import AutoTokenizer
    a, b = [AutoTokenizer.from_pretrained(p, local_files_only=True, trust_remote_code=False) for p in (axonai,qwen)]
    rows = read_rows(root)
    for row in rows:
        packet=[{'role':'system','content':SYSTEM},{'role':'user','content':row['prompt']}]
        kwargs=dict(tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        ids=a.apply_chat_template(packet,**kwargs)
        assert ids == b.apply_chat_template(packet,**kwargs), 'Cross-arm tokenizer mismatch'
        row.update(input_tokens=len(ids),input_ids_sha256=ids_digest(ids),
                   eligible=len(ids)<=2048 and '<|im_start|>' not in row['prompt'] and '<|im_end|>' not in row['prompt'])
        row['exclusion'] = None if row['eligible'] else 'context_overflow' if len(ids)>2048 else 'reserved_delimiter'
    # Interleave suites by seeded whole-roster shuffle; four equal blocks, AB/BA/AB/BA.
    ordered = list(rows); random.Random(42).shuffle(ordered)
    for i,row in enumerate(ordered):
        row['block'] = min(3, i*4//len(ordered))
        row['order'] = i
    destination=root/'roster.jsonl'
    with destination.open('x') as f:
        for row in ordered: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    summary={'planned':SUITES,'gpqa':'pending_user_arranged_access','arms':['qwen','axonai'],
             'eligible':dict(Counter(r['suite'] for r in rows if r['eligible'])),
             'excluded':dict(Counter(r['suite'] for r in rows if not r['eligible'])),
             'roster_sha256':sha(destination),'ordered_ids_sha256':hashlib.sha256(canonical([r['id'] for r in ordered])).hexdigest(),
             'renderer_sha256':sha(__file__),'input_limit':2048,'output_limit':256,'seed':42,
             'block_arm_order':[['qwen','axonai'],['axonai','qwen'],['qwen','axonai'],['axonai','qwen']],
             'source_files':{str(p.relative_to(root)):sha(p) for p in sorted((root/'sources').rglob('*')) if p.is_file()}}
    (root/'roster-freeze.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('planned','eligible','excluded','roster_sha256')},indent=2))


def paired_stats(pairs, seed=42):
    import numpy as np
    if not pairs:
        return {'jointly_returned':0}
    values=np.array(pairs,dtype=np.int8)
    difference=values[:,1]-values[:,0]  # axonai minus qwen
    rng=np.random.default_rng(seed)
    bootstrap=np.empty(10000)
    for i in range(0,10000,100):
        bootstrap[i:i+100]=difference[rng.integers(0,len(values),(100,len(values)))].mean(axis=1)*100
    wins=int((difference==1).sum()); losses=int((difference==-1).sum()); n=wins+losses
    # Exact two-sided binomial test; log-space avoids huge combinatorial intermediates.
    k=min(wins,losses)
    probability=1.0 if n==0 else min(1.,2*sum(math.exp(math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1)-n*math.log(2)) for j in range(k+1)))
    return {'jointly_returned':len(pairs),'axonai_wins':wins,'qwen_wins':losses,
            'both_correct':int(((values[:,0]==1)&(values[:,1]==1)).sum()),
            'both_incorrect':int(((values[:,0]==0)&(values[:,1]==0)).sum()),
            'difference_percentage_points':float(difference.mean()*100),
            'paired_bootstrap_95_percent':np.percentile(bootstrap,[2.5,97.5]).tolist(),
            'exact_mcnemar_two_sided_p':probability}


def analyze(root):
    import sqlite3
    import numpy as np
    rows=[json.loads(line) for line in (root/'roster.jsonl').read_text().splitlines()]
    db=sqlite3.connect(root/'attempts.sqlite3')
    outcomes={(ident,arm):json.loads(payload) for ident,arm,payload in db.execute("SELECT item_id,arm,outcome FROM attempts WHERE outcome IS NOT NULL")}
    result={'roster_sha256':sha(root/'roster.jsonl'),'primary':'pubmedqa','gpqa':'pending_access',
            'overlap_screened_sensitivity':'unavailable: incomplete corpus and family-crosswalk coverage',
            'scorer_sha256':sha(__file__),'suites':{}}
    for suite,total in SUITES.items():
        selected=[r for r in rows if r['suite']==suite and r['eligible']]
        summary={'scheduled':total,'eligible':len(selected),'excluded':total-len(selected),'arms':{}}
        correctness={}; returned={}
        for arm in ('qwen','axonai'):
            correct=parsed=0;returned[arm]=set();correctness[arm]={};latency=[];tokens=[];cap=0;errors=Counter();confusion=Counter();subjects=defaultdict(Counter);peaks=[0,0]
            for row in selected:
                out=outcomes.get((row['id'],arm))
                if out is None: errors['missing']+=1;continue
                if out.get('status') not in ('ok','empty_output'):
                    errors[out.get('status','unknown')]+=1;continue
                returned[arm].add(row['id'])
                guess=parse_answer(out.get('text',''),suite);ok=guess==row['gold'];parsed+=guess is not None;correct+=ok
                correctness[arm][row['id']]=ok;confusion[row['gold']+'->'+str(guess)]+=1
                subjects[row.get('subject','all')]['correct']+=ok;subjects[row.get('subject','all')]['returned']+=1
                latency.append(out['elapsed_s']);tokens.append(out['output_tokens']);cap+=out['stop']=='cap'
                for i,gpu in enumerate(out.get('gpu',[])):
                    peaks[i]=max(peaks[i],gpu.get('reserved',gpu.get('reserved_bytes',0)))
            arm_result={'returned':len(returned[arm]),'parsed':parsed,'correct':correct,
                        'operational_accuracy':correct/len(selected) if selected else None,
                        'infra_failures':dict(errors),'cap_hits':cap,'output_tokens':sum(tokens),
                        'latency_seconds_p50_p95':np.percentile(latency,[50,95]).tolist() if latency else None,
                        'peak_reserved_device_bytes':peaks,'subject_counts':dict(subjects)}
            if suite=='pubmedqa':
                labels=('yes','no','maybe');f1=[]
                for label in labels:
                    tp=confusion[label+'->'+label];fp=sum(v for k,v in confusion.items() if k.endswith('->'+label) and not k.startswith(label+'->'));fn=sum(v for k,v in confusion.items() if k.startswith(label+'->') and k!=label+'->'+label)
                    f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
                arm_result.update(confusion=dict(confusion),macro_f1=sum(f1)/3)
            summary['arms'][arm]=arm_result
        pairs=[(correctness['qwen'][r['id']],correctness['axonai'][r['id']]) for r in selected if r['id'] in returned['qwen']&returned['axonai']]
        summary['paired']=paired_stats(pairs)
        # Bounds allow either result for every missing arm outcome over the eligible roster.
        observed={a:sum(correctness[a].values()) for a in correctness};missing={a:len(selected)-len(returned[a]) for a in returned}
        n=len(selected)
        summary['difference_bounds_percentage_points']=[100*(observed['axonai']-observed['qwen']-missing['qwen'])/n,100*(observed['axonai']+missing['axonai']-observed['qwen'])/n] if n else None
        if suite=='pubmedqa':summary['majority_baseline']=max(Counter(r['gold'] for r in selected).values())/n
        if suite!='pubmedqa':summary['paired'].pop('exact_mcnemar_two_sided_p',None)
        result['suites'][suite]=summary
    result['attempt_states']=dict(db.execute('SELECT state,count(*) FROM attempts GROUP BY state'))
    destination=root/'results.json';temporary=destination.with_suffix('.json.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(destination)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','analyze']);parser.add_argument('root',type=Path);parser.add_argument('--axonai',type=Path);parser.add_argument('--qwen',type=Path);args=parser.parse_args()
    if args.command=='prepare':prepare(args.root,args.axonai,args.qwen)
    else:analyze(args.root)
