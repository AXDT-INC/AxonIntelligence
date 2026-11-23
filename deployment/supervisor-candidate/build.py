"""Offline GCC-only build. Identical repeat builds; writes only this candidate."""
import os,sys,subprocess,tempfile,json,hashlib
from pathlib import Path
ROOT=Path(__file__).absolute().parent
FLAGS=['-pipe','-std=c11','-O2','-mno-f16c','-fno-fast-math','-ffp-contract=off','-fexcess-precision=standard','-Wall','-Wextra','-Werror','-Wno-misleading-indentation','-Wl,-z,now,-z,relro']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 env={'PATH':'/usr/bin:/bin','LC_ALL':'C','TMPDIR':str(ROOT/'build-tmp')}
 outputs={'worker':['worker.c','convert.c','verify.c'],'supervisor':['supervisor.c'],'harness':['harness.c'],'exhaustive':['tests/exhaustive.c','convert.c','verify.c'],'kernel':['tests/kernel.c','convert.c','verify.c']}
 hashes={}
 with tempfile.TemporaryDirectory(dir=ROOT/'build-tmp') as d:
  for name,sources in outputs.items():
   built=[]
   for i in range(2):
    path=Path(d)/(name+str(i));subprocess.run(['/usr/bin/gcc',*FLAGS,'-I.',*sources,'-lm','-o',str(path)],cwd=ROOT,env=env,check=True);built.append(path.read_bytes())
   if built[0]!=built[1]:raise RuntimeError('nonreproducible local build')
   fd=os.open(ROOT/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o500)
   with os.fdopen(fd,'wb') as f:f.write(built[0])
   hashes[name]=hashlib.sha256(built[0]).hexdigest()
 sources={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.suffix in ('.py','.c','.h') and 'test-workspace' not in p.parts and 'build-tmp' not in p.parts}
 trust={'abi':2,'sources':sources,'executables':hashes,'compiler':subprocess.check_output(['/usr/bin/gcc','--version'],env=env,text=True).splitlines()[0],'compiler_sha256':sha(Path('/usr/bin/gcc')),'flags':FLAGS,'repeat_build_identical':True,'system_dependencies':'Linux/glibc/dynamic linker and Python are trusted system dependencies, not sealed.'}
 with (ROOT/'trust.json').open('x') as f:json.dump(trust,f,sort_keys=True,indent=2);f.write('\n')
 (ROOT/'trust.json').chmod(0o644)
 print(json.dumps(trust,sort_keys=True))
if __name__=='__main__':main()
