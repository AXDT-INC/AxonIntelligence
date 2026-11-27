"""Screen public items against available frozen/staging text; explicitly retain gaps.

Sparse arithmetic implements the v1 lexical thresholds, not a clean-set clearance.
Historical sealed evaluations are never opened.
"""
import json
from pathlib import Path
import re
import sys
from collections import Counter,defaultdict
import numpy as np

CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE.parent/'v1'))
from screen import features
from public_suite import sha


def identifiers(row):
    p=row.get('provenance',{});ids=set()
    for field in ['pmid','pmcid','doi','source_id','source_document_id','source_unit_id']:
        value=str(p.get(field,'')).strip()
        if not value or value=='None':continue
        if field=='pmid' and value.isdigit():ids.add('pmid:'+value)
        ids.update('pmcid:'+m.upper() for m in re.findall(r'PMC\d+',value,re.I))
        if field=='doi':ids.add('doi:'+value.casefold().removeprefix('https://doi.org/'))
    return ids


def matrix(feature_sets,vocab):
    from scipy.sparse import csr_matrix
    columns=[];offsets=[0]
    for items in feature_sets:
        columns.extend(sorted(vocab[x] for x in items if x in vocab));offsets.append(len(columns))
    return csr_matrix((np.ones(len(columns),dtype=np.int32),columns,offsets),shape=(len(feature_sets),len(vocab)))


def lexical_matches(query_features,reference_features):
    # Testable arithmetic shared by real scanning and planted fixtures.
    qw=[x[1] for x in query_features];qt=[x[2] for x in query_features]
    vocabw={x:i for i,x in enumerate(set().union(*qw))};vocabt={x:i for i,x in enumerate(set().union(*qt))}
    return matrix(qw,vocabw).T.tocsr(),matrix(qt,vocabt).T.tocsr(),vocabw,vocabt


def main(root):
    sys.path.insert(0,str(root/'screen-tools'))
    rows=[json.loads(l) for l in (root/'roster.jsonl').read_text().splitlines()]
    facets=[];owners=[];candidate_sources={r['family']:r['id'] for r in rows}
    for row in rows:
        for text in [row['question'],*row.get('contexts',[])]:
            facets.append(features(text));owners.append(row['id'])
    qw,qt,vocabw,vocabt=lexical_matches(facets,[])
    nqw=np.array([len(x[1]) for x in facets]);nqt=np.array([len(x[2]) for x in facets])
    exact=defaultdict(set)
    for feat,ident in zip(facets,owners):exact[feat[0]].add(ident)
    result={r['id']:{'signals':set(),'source_matches':set()} for r in rows}
    repo=CODE.parents[2]
    corpora=[('frozen_sft',repo/'engineering-maintenance/frozen-v1-training-extraction-v1-20281029/train-10684.jsonl','641b43e32f3d69fa6fbf74754772e976777a26383c27de2692b90670c5dc5d07',10684),
             ('staging',repo/'datasets/axonai-v1/generated/20k-completion-production/run-v1/recovery/final-20000-staging-v11-20261009T051315Z/axonai-v1-20000-STAGING-NOT-FROZEN.jsonl','24889f6495d4e9e5015f7a63352fba470b9913fcb1dbeebc7d84c0eb0546b5fd',20000)]
    inventory=[];alias=defaultdict(set)
    # Build only known source crosswalks from authorized frozen rows.
    with corpora[0][1].open() as f:
        for line in f:
            ids=identifiers(json.loads(line))
            for ident in ids:alias[ident].update(ids)
    def compare(batch):
        if not batch:return
        f=[features(text) for text in batch];rw=[x[1] for x in f];rt=[x[2] for x in f]
        sharedw=(matrix(rw,vocabw)@qw).toarray();sharedt=(matrix(rt,vocabt)@qt).toarray()
        nrw=np.array([len(x) for x in rw])[:,None];nrt=np.array([len(x) for x in rt])[:,None]
        jac=sharedw/np.maximum(1,nrw+nqw-sharedw)
        containment=sharedt/np.maximum(1,np.minimum(nrt,nqt))
        rich=np.minimum(nrw,nqw)>=6
        probable=rich&((jac>=.65)|((sharedt>=3)&(containment>=.65)))
        unresolved=rich&((jac>=.48)|((sharedt>=3)&(containment>=.45)))&~probable
        for name,mask in [('probable_paraphrase',probable),('unresolved_lexical',unresolved)]:
            for j in np.flatnonzero(mask.any(axis=0)):result[owners[j]]['signals'].add(name)
        for canon,_,_ in f:
            for ident in exact.get(canon,()):result[ident]['signals'].add('exact_match')
    for role,path,pin,count in corpora:
        assert sha(path)==pin
        seen=0;batch=[]
        with path.open() as stream:
            for line in stream:
                row=json.loads(line);seen+=1;ids=identifiers(row)
                ids=ids|set().union(*(alias[x] for x in ids)) if ids else ids
                for source in ids:
                    if source in candidate_sources:
                        r=result[candidate_sources[source]];r['signals'].add('shared_source');r['source_matches'].add(source)
                batch.extend(m['content'] for m in row['messages'] if m.get('role') in ('user','assistant') and isinstance(m.get('content'),str))
                if len(batch)>=128:compare(batch);batch=[]
                if seen%2000==0:print(json.dumps({'role':role,'records':seen}),flush=True)
        compare(batch);assert seen==count
        inventory.append({'role':role,'path':str(path),'sha256':pin,'rows':seen})
    for value in result.values():
        value['signals']=sorted(value['signals']);value['source_matches']=sorted(value['source_matches']);value['coverage']='incomplete'
    receipt={'roster_sha256':sha(root/'roster.jsonl'),'implementation_sha256':sha(__file__),
             'threshold_source_sha256':sha(CODE.parent/'v1/screen.py'),'corpora':inventory,
             'missing_coverage':['candidate_corpora_not_exhaustively_inventoried','generated_variants_not_exhaustively_inventoried','previous_evaluations_not_inventoried','full_PMID_PMCID_DOI_and_source_family_crosswalk_unverified','historical_sealed_51_unassessed'],
             'clean_sensitivity_available':False,'disposition':'Full public-set evaluation only; no clean/generalization claim.',
             'signals_by_item':result,'signal_counts':dict(Counter(s for r in result.values() for s in r['signals']))}
    with (root/'screening.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps({'complete':True,'signal_counts':receipt['signal_counts'],'clean_sensitivity_available':False}),flush=True)


if __name__=='__main__':main(Path(sys.argv[1]).resolve())
