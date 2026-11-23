import json,statistics,collections
from pathlib import Path
P=Path(__file__).resolve().parent

def edit_distance(a,b):
 old=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  new=[i]
  for j,y in enumerate(b,1):new.append(min(new[-1]+1,old[j]+1,old[j-1]+(x!=y)))
  old=new
 return old[-1]

def compare():
 baseline=[json.loads(x) for x in (P/'baseline/responses.jsonl').read_text().splitlines()];merged=[json.loads(x) for x in (P/'merged-repair2/responses.jsonl').read_text().splitlines()];assert len(baseline)==len(merged)==6
 results=[]
 for a,b in zip(baseline,merged):
  assert a['id']==b['id'] and a['rendered_sha256']==b['rendered_sha256'] and a['input_ids']==b['input_ids'];x=a['output_ids'];y=b['output_ids'];prefix=0
  for u,v in zip(x,y):
   if u!=v:break
   prefix+=1
  results.append({'id':a['id'],'prompt_token_ids_identical':True,'exact_generated_token_equality':x==y,'exact_text_equality':a['text']==b['text'],'baseline_output_tokens':len(x),'merged_output_tokens':len(y),'common_prefix_tokens':prefix,'token_edit_distance':edit_distance(x,y),'baseline_text':a['text'],'merged_text':b['text'],'baseline_token_ids':x,'merged_token_ids':y,'baseline_latency_seconds':a['latency_seconds'],'merged_latency_seconds':b['latency_seconds'],'baseline_truncated':a['truncated'],'merged_truncated':b['truncated'],'teacher_forced_logits':b['aligned_baseline_prefix_logits']})
 totals={'comparison_prompts':6,'identical_prompt_tokenizations':6,'exact_token_matches':sum(x['exact_generated_token_equality'] for x in results),'exact_text_matches':sum(x['exact_text_equality'] for x in results),'baseline_output_tokens':sum(x['baseline_output_tokens'] for x in results),'merged_output_tokens':sum(x['merged_output_tokens'] for x in results),'token_edit_distance_sum':sum(x['token_edit_distance'] for x in results),'aligned_logits_positions':sum(x['teacher_forced_logits']['positions'] for x in results),'aligned_logits_top1_agreement':sum(x['teacher_forced_logits']['top1_agreement'] for x in results),'maximum_logit_abs_difference':max(x['teacher_forced_logits']['max_abs_difference'] for x in results),'truncations':sum(x['baseline_truncated']+x['merged_truncated'] for x in results),'bitwise_equivalence_demonstrated':False,'note':'Token edit distance includes insertions/deletions/substitutions; aligned logits use the same first up-to-32 baseline-generated prefix positions. Differences beyond those positions are not fully diagnosed by this statistic.'}
 (P/'TOKEN_COMPARISON.json').write_text(json.dumps({'summary':totals,'rows':results},indent=2)+'\n')
 print(json.dumps(totals,indent=2))
 for r in results:print(r['id'],'BASELINE:',r['baseline_text'],'MERGED:',r['merged_text'])
if __name__=='__main__':compare()
