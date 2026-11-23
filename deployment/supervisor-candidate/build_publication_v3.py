"""Rebuild only the supervisor, twice, with existing compiler/flags; no installation.
Historical sources are not added to the live executable trust set.
"""
import hashlib,json,os,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).absolute().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 trust=json.loads((ROOT/'trust.json').read_text())
 assert sha(Path('/usr/bin/gcc'))==trust['compiler_sha256']
 env={'PATH':'/usr/bin:/bin','LC_ALL':'C','TMPDIR':str(ROOT/'build-tmp')}
 with tempfile.TemporaryDirectory(dir=ROOT/'build-tmp') as temp:
  paths=[Path(temp)/('supervisor'+str(i)) for i in range(2)]
  for p in paths:subprocess.run(['/usr/bin/gcc',*trust['flags'],'-I.','supervisor.c','-lm','-o',str(p)],cwd=ROOT,env=env,check=True)
  assert paths[0].read_bytes()==paths[1].read_bytes()
  paths[0].chmod(0o500);os.replace(paths[0],ROOT/'supervisor')
 for name in trust['sources']:trust['sources'][name]=sha(ROOT/name)
 for name in ('build_publication_v3.py','tests/test_publication_gate_v3.py','run_publication_v3.py','tests/commit_process_cases.py'):
  if (ROOT/name).exists():trust['sources'][name]=sha(ROOT/name)
 trust['executables']['supervisor']=sha(ROOT/'supervisor');trust['commit_protocol']=3
 (ROOT/'trust.json').write_text(json.dumps(trust,sort_keys=True,indent=2)+'\n')
 print(json.dumps({'supervisor_sha256':trust['executables']['supervisor'],'repeat_build_identical':True}))
if __name__=='__main__':main()
