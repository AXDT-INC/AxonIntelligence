import hashlib,json,os,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).absolute().parent
FLAGS=['-std=c11','-O2','-fPIC','-shared','-pthread','-Wall','-Wextra','-Werror','-Wno-misleading-indentation','-Wl,-z,now,-z,relro']
def main():
 t=json.loads((ROOT/'trust.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 assert sha(Path('/usr/bin/gcc'))==t['compiler_sha256']
 with tempfile.TemporaryDirectory(dir=ROOT/'build-tmp') as temp:
  paths=[Path(temp)/str(i) for i in range(2)]
  for p in paths:subprocess.run(['/usr/bin/gcc',*FLAGS,'publication_gate.c','-o',str(p)],cwd=ROOT,env={'PATH':'/usr/bin:/bin','LC_ALL':'C','TMPDIR':str(ROOT/'build-tmp')},check=True)
  assert paths[0].read_bytes()==paths[1].read_bytes();paths[0].chmod(0o500);os.replace(paths[0],ROOT/'publication_gate.so')
 for name in list(t['sources'])+['build_atomic_publication.py','publication_gate.c','publication_gate.so','tests/test_atomic_publication.py']:
  if (ROOT/name).exists():t['sources'][name]=sha(ROOT/name)
 t['publication_native_build']={'flags':FLAGS,'identical_repeat':True,'sha256':sha(ROOT/'publication_gate.so')}
 (ROOT/'trust.json').write_text(json.dumps(t,indent=2,sort_keys=True)+'\n')
 print(json.dumps(t['publication_native_build']))
if __name__=='__main__':main()
