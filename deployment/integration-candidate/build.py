"""Offline build into a fresh private directory; never follows existing output paths."""
import os,json,hashlib,tempfile,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
FLAGS=['-pipe','-std=c11','-O2','-mno-f16c','-fno-fast-math','-ffp-contract=off','-fexcess-precision=standard','-Wall','-Wextra','-Werror','-Wl,-z,now,-z,relro']
SOURCES=['worker.c','convert.c','verify.c','prototype.h']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if __name__=='__main__':
 env={'PATH':'/usr/bin:/bin','LC_ALL':'C','TMPDIR':str(ROOT/'build-tmp')}
 with tempfile.TemporaryDirectory(dir=ROOT/'build-tmp') as d:
  binaries=[]
  for i in range(2):
   out=Path(d)/str(i)
   subprocess.run(['/usr/bin/gcc',*FLAGS,*SOURCES[:3],'-lm','-o',str(out)],cwd=ROOT,env=env,check=True)
   binaries.append(out.read_bytes())
  assert binaries[0]==binaries[1]
  manifest={'abi':1,'sources':{n:sha(ROOT/n) for n in SOURCES},'compiler':subprocess.check_output(['/usr/bin/gcc','--version'],env=env,text=True).splitlines()[0],'compiler_sha256':sha(Path('/usr/bin/gcc')),'flags':FLAGS,'executable_sha256':hashlib.sha256(binaries[0]).hexdigest(),'repeat_build_identical':True}
  fd=os.open(ROOT/'worker',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o500)
  with os.fdopen(fd,'wb') as f:f.write(binaries[0])
  with (ROOT/'trust.json').open('x') as f:json.dump(manifest,f,sort_keys=True,indent=2);f.write('\n')
 print(json.dumps(manifest))
