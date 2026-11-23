"""Synthetic end-to-end measurements; reports remain in candidate only."""
import os,sys,time,json,resource,statistics,tempfile,subprocess
from pathlib import Path
ROOT=Path(__file__).absolute().parent
sys.path.insert(0,str(ROOT/'tests'))
from test_conversion import ConversionTests
import candidate,native_engine as n,convert_fp16 as c
from unittest.mock import patch
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
if os.nice(0)<19:os.nice(19-os.nice(0))
rows=[];workers=[];finish=n.Native.finish
def observe(w):
 if not w.closed:
  mem={s.split(':')[0]:s.split(':')[1].strip() for s in Path('/proc/'+str(w.proc.pid)+'/status').read_text().splitlines() if s.startswith(('VmHWM','VmRSS'))}
  workers.append({'startup_seconds':w.startup_seconds,'memory':mem})
 return finish(w)
with patch.object(n.Native,'finish',observe):
 fixture=ConversionTests();fixture.setUp()
 try:
  # Mixed normal/subnormal values, identical payload and headers for each engine.
  bits=[0x3f80,0xbf80,0x3eaa,0x3880,0x3300,0x3301,0,0x8000]*131072
  fixture.set_tensors([('data',bits)]);del bits
  for trial in range(3):
   for selection in ('reference','native'):
    row={'trial':trial,'engine':selection,'payload_bytes':2097152}
    for phase in ('convert_with_internal_verification','independent_verification'):
     cpu=time.process_time();child=resource.getrusage(resource.RUSAGE_CHILDREN);wall=time.perf_counter()
     if phase.startswith('convert'):candidate.convert(str(fixture.source),str(fixture.parent),selection+str(trial),fixture.anchor,selection=selection)
     else:candidate.verify(str(fixture.source),str(fixture.parent/(selection+str(trial))),fixture.anchor,selection=selection)
     dt=time.perf_counter()-wall;after=resource.getrusage(resource.RUSAGE_CHILDREN)
     totalcpu=time.process_time()-cpu+after.ru_utime+after.ru_stime-child.ru_utime-child.ru_stime
     row[phase]={'seconds':dt,'MiB_s':2/dt,'cpu_percent_combined':100*totalcpu/dt}
    rows.append(row)
    print(json.dumps(row),flush=True);time.sleep(.1)
 finally:fixture.doCleanups()
# Compare protocol with same C routines executed locally in a fresh test subprocess.
with tempfile.TemporaryDirectory(dir=ROOT/'test-workspace') as td:
 t=Path(td);src=r'''
#include "prototype.h"
#include <stdio.h>
#include <time.h>
int main(void){unsigned char a[65536],b[65536];unsigned words[]={0x3f80,0xbf80,0x3eaa,0x3880,0x3300,0x3301,0,0x8000};for(unsigned i=0;i<32768;i++){a[2*i]=words[i%8];a[2*i+1]=words[i%8]>>8;}for(int phase=0;phase<2;phase++){Statistics s={0};clock_t start=clock();for(int i=0;i<32;i++){int rc=phase?prototype_verify(a,sizeof a,b,sizeof b,&s):prototype_convert(a,sizeof a,b,sizeof b,&s);if(rc)return 1;}printf("%.9f\n",(double)(clock()-start)/CLOCKS_PER_SEC);}return 0;}
'''
 (t/'kernel.c').write_text(src)
 subprocess.run(['/usr/bin/gcc','-O2','-fno-fast-math','-ffp-contract=off','-I',str(ROOT),str(t/'kernel.c'),str(ROOT/'convert.c'),str(ROOT/'verify.c'),'-lm','-o',str(t/'kernel')],env={'PATH':'/usr/bin:/bin','TMPDIR':td},check=True)
 kernel=[float(x) for x in subprocess.check_output([str(t/'kernel')],text=True).splitlines()]
 raw=b''.join(x.to_bytes(2,'little') for x in [0x3f80,0xbf80,0x3eaa,0x3880,0x3300,0x3301,0,0x8000])*8192
 with candidate.engine('native') as w:
  st=c.empty_stats();start=time.perf_counter()
  for _ in range(16):out=w.block(raw,st)
  convert_time=time.perf_counter()-start;st=c.empty_stats();start=time.perf_counter()
  for _ in range(16):w.block(raw,st,out)
  verify_time=time.perf_counter()-start
result={'payload_bytes':2097152,'trials':rows,'workers':workers,'coordinator_memory':{s.split(':')[0]:s.split(':')[1].strip() for s in Path('/proc/self/status').read_text().splitlines() if s.startswith(('VmHWM','VmRSS'))},'kernel_seconds':kernel,'supervised_block_seconds':[convert_time,verify_time],'ipc_and_wrapper_overhead_seconds':[convert_time-kernel[0],verify_time-kernel[1]],'notes':'One CPU affinity, nice19. Combined RSS upper bound is sum of separately observed process high-water marks, not simultaneous RSS. CPU includes child accounting. 2MiB fixtures, warm cache, synthetic file fsync and hashes included; external receipt publication excluded.'}
result['projection_minutes']={engine:53791996928/2097152*sum(statistics.median(r[ph]['seconds'] for r in rows if r['engine']==engine) for ph in ['convert_with_internal_verification','independent_verification'])/60 for engine in ['reference','native']}
with (ROOT/'benchmark-results.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(result),flush=True)
