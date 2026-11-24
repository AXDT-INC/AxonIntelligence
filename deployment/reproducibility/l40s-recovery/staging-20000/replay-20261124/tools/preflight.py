"""New bounded CPU-only preflight; does not import or execute assembler."""
import ast
import ctypes
import hashlib
import json
import os
import platform
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/data/axonai/axonai-v1-original-assembler-replay-20261124')
DATA = Path('/data/axonai/datasets/axonai-v1')
HIST = DATA/'generated/20k-completion-production/run-v1/recovery/final-20000-staging-v11-20261009T051315Z'
PIN_SOURCE = 'cbbb8750b0d21d129899578cfbf61ce7b1a8af665f2761277e70c55b3a9b36f9'
PINS = {
 'batch001': ('8f3918f4eece4e83e87702308a972b1a620e76f34d0d244bb78a8df7792dfd67', DATA/'sft/train/repaired/batch001-repaired-20261002T141416Z/batch001-5800-repaired.jsonl'),
 'pmc': ('4ebc148178d70bd65302bbdc7714e3bde6e5fca662a08cc71a1dc1bc590d7f7f', DATA/'sft/train/pmc-batch002-7000.jsonl'),
 'production': ('ba0f5b5b20add36c459bdda395ec2311626c7aeb0de33b553f9fcf24bce563f4', DATA/'generated/20k-completion-production/run-v1/accepted-canonical.jsonl'),
 'recovered28_archive': ('c479afe31be7c1212e41100fbdc274a4a74c00603c9beb4a553a24a93a7a2f46', ROOT/'inputs/axonai-reconcile-28-20261008T155441Z.tar.gz'),
 'supersession_plan': ('b73c01129e0fb30939fca75422ba80925582ff7ce261f9334faba1919c622d5e', ROOT/'inputs/axonai-duplicate-replacement-plan-20261008T163702Z.tar.gz'),
 'previous11_archive': ('3e88d4599b2c5b967d67ebc1bc15d54b8a42bfbc4fc6a9f22bcd389b824c527c', ROOT/'inputs/axonai-duplicate12-generation-20261009T042944Z.tar.gz'),
 'single_archive': ('bcc4edb9dd34b0f76545bd0cd190568419faff46d117cbb18d8c59e01d95acee', ROOT/'inputs/axonai-slot06926-generation-20261009T045823Z.tar.gz'),
 'source12_archive': ('7a99de30ab0a60f73d9c80e954e0021f568e0048d919c8c9c0f460e5655a1fd5', ROOT/'inputs/axonai-duplicate12-source-freeze-20261009T042202Z.tar.gz'),
 'source1_archive': ('aebfe44772bf22a74a8b913590a8138ff96de973565c71b3e38143bb7fb2a955', ROOT/'inputs/axonai-slot06926-source-freeze-20261009T044828Z.tar.gz'),
}
OUTPUT_PINS = {
 'axonai-v1-20000-STAGING-NOT-FROZEN.jsonl':'24889f6495d4e9e5015f7a63352fba470b9913fcb1dbeebc7d84c0eb0546b5fd',
 'input-sha256.json':'0a86d4cf03345b9ab3e361e5547f3295c030d07ea99a2ecf6e5fab65592d5312',
 'near-question-review.json':'06eabd8fd70ecf7a9cfd9e8b068afcb412f1a691bfb041b7243b5b26567d5bcb',
 'staging-audit-report.json':'0d447cee999af9271837796544c9fccfcca1089d6e391734cefb1838ad92f860',
 'superseded-production-record-ids.json':'d332eab16ccb587ca7897ae0defab0d8df8f74e8e50a65ad9e08e3e65011d272',
}

def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def identity(p):
    s=p.stat()
    return dict(path=str(p),bytes=s.st_size,sha256=digest(p),mtime_ns=s.st_mtime_ns,
                inode=s.st_ino,device=s.st_dev,mode=oct(s.st_mode & 0o777),links=s.st_nlink)

def main():
    assert sys.version_info[:3] == (3,12,3)
    assert not (ROOT/'execution/output').exists()
    for p in [ROOT, ROOT/'inputs', ROOT/'execution', ROOT/'tmp', ROOT/'logs', ROOT/'tools']:
        assert p.resolve()==p and p.is_dir() and not p.is_symlink()
        assert p.stat().st_mode & 0o077 == 0, str(p)
    historical=json.loads((HIST/'input-sha256.json').read_text())
    assert historical == {k:v[0] for k,v in PINS.items()}
    src=ROOT/'inputs/axonai_stage_final_20000_v11.py'
    files={}
    for key,(pin,p) in {'assembler':(PIN_SOURCE,src),**PINS}.items():
        assert p.is_file() and p.resolve()==p and not p.is_symlink()
        info=identity(p)
        assert info['sha256']==pin, key
        files[key]=info
    source=src.read_bytes(); tree=ast.parse(source);compile(source,str(src),'exec')
    imports=sorted({a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names} |
                   {n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)})
    assert set(imports)<=set(sys.stdlib_module_names)|{'__future__'}
    archive_checks=[]
    for key,(pin,p) in PINS.items():
        if p.suffix!='.gz':continue
        with tarfile.open(p,'r:gz') as tf:
            members=tf.getmembers()
            assert len(members)<1000 and sum(m.size for m in members)<64*1024**2
            assert all(m.isfile() or m.isdir() for m in members)
            assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
            regular=[m for m in members if m.isfile()]
            basenames=[Path(m.name).name for m in regular]
            assert len(set(basenames))==len(basenames)
            byname=dict(zip(basenames,regular))
            checks=tf.extractfile(byname['SHA256SUMS']).read().decode().splitlines()
            for line in checks:
                pin,name=line.strip().split(None,1)
                content=tf.extractfile(byname[Path(name.strip()).name]).read()
                assert hashlib.sha256(content).hexdigest()==pin
            archive_checks.append(dict(input=key,members=len(members),regular_files=len(regular),
                expanded_bytes=sum(m.size for m in regular),internal_checksums=len(checks),status='PASS'))
    references={name:identity(HIST/name) for name in OUTPUT_PINS}
    assert all(references[n]['sha256']==h for n,h in OUTPUT_PINS.items())
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    disk=shutil.disk_usage(ROOT)
    assert available>16*1024**3 and disk.free>1024**3
    libc=ctypes.CDLL(None,use_errno=True);abi=libc.syscall(444,0,0,1);assert abi>=4
    cgroup=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split('::')[1].lstrip('/')
    limits={}
    for p in [cgroup,*cgroup.parents]:
        if p==Path('/sys/fs'):break
        limit=p/'memory.max'
        if limit.exists():
            value=limit.read_text().strip();limits[str(limit)]=value
            assert value=='max' or int(value)>16*1024**3
    result=dict(status='PASS',time_utc=datetime.now(timezone.utc).isoformat(),files=files,
      imports=imports,archive_checks=archive_checks,historical_outputs=references,
      runtime=dict(python=sys.version,executable=sys.executable,prefix=sys.prefix,platform=platform.platform(),
      mem_available_bytes=available,disk_free_bytes=disk.free,cgroup_memory_limits=limits,landlock_abi=abi,
      replay_address_space_limit_bytes=8*1024**3),
      static_review=dict(complete_source_reviewed=True,third_party_dependencies=[],gpu_imports=False,
      network_calls=False,subprocess_calls=False,archive_extraction_to_disk=False,
      archive_access='tarfile r:gz, regular members read with extractfile; no extraction to filesystem',
      writes='output_dir mkdir; five output files via temporary files, fsync and os.replace; output SHA256SUMS',
      original_paths_opened_for_write=False,near_screen='unchanged default enabled, no skip flag'),
      output=str(ROOT/'execution/output'))
    with (ROOT/'logs/preflight.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':'PASS','verified_source_and_inputs':len(files),'archive_internal_checks':archive_checks,
                      'memory_available_bytes':available,'disk_free_bytes':disk.free,'landlock_abi':abi}))

if __name__=='__main__':main()
