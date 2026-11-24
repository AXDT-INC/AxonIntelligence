"""Test the new confinement harness using disposable replay-local files only."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

root=Path('/data/axonai/axonai-v1-original-assembler-replay-20261124')
python='/data/axonai/.venv/bin/python'
canary=root/'logs/write-protection-canary.txt'
with canary.open('x') as f:f.write('Replay-only write protection canary.\n')
before=hashlib.sha256(canary.read_bytes()).hexdigest()
code='''
import json, os, pathlib
root=pathlib.Path('/data/axonai/axonai-v1-original-assembler-replay-20261124')
try:
    with (root/'logs/write-protection-canary.txt').open('wb') as f:
        raise RuntimeError('Write restriction failed')
except PermissionError:
    pass
for d in ['execution','tmp']:
    p=root/d/'confinement-probe'
    p.mkdir()
    (p/'a').write_text('probe')
    os.replace(p/'a',p/'b')
    assert (p/'b').read_text()=='probe'
    (p/'b').unlink();p.rmdir()
print(json.dumps({'status':'PASS','outside_allowlist_write_denied':True,
    'create_write_replace_read_unlink_rmdir_inside_allowlist_passed':True}))
'''
cmd=[python,'-I','-B',str(root/'tools/restrict_writes.py'),str(root/'execution'),str(root/'tmp'),
     '--',python,'-I','-B','-c',code]
proc=subprocess.run(cmd,capture_output=True,text=True,check=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''})
assert before==hashlib.sha256(canary.read_bytes()).hexdigest()
result=json.loads(proc.stdout)
result['protected_canary_preserved']=True
result['note']='Only disposable replay-local files tested; no write attempt against original artifacts.'
with (root/'logs/confinement-check.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
