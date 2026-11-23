"""Small warm-cache end-to-end benchmark; never opens production weights."""
import time,json,resource,statistics,os
from unittest.mock import patch
from pathlib import Path
from test_conversion import ConversionTests
import candidate,native_engine as n,convert_fp16 as c
ROOT=Path(__file__).absolute().parent

def main():
 fixture=ConversionTests();fixture.setUp();rows=[];startups=[];observations=[];finish=n.Native.finish
 def observed(w):
  before=w.info.get('supervisor_cpu_seconds',0)
  if not w.closed:
   rss={}
   for label,pid in [('coordinator',os.getpid()),('supervisor',os.getppid()),('worker',w.info['pid'])]:
    try:rss[label]=int(next(line.split()[1] for line in Path('/proc/'+str(pid)+'/status').read_text().splitlines() if line.startswith('VmRSS:')))
    except (OSError,StopIteration):rss[label]=0
   finish_start=time.perf_counter();finish(w);cleanup_seconds=time.perf_counter()-finish_start;observations.append(dict(w.info,cleanup_seconds=cleanup_seconds,combined_sample_rss_kib=sum(rss.values()),rss_sample=rss,supervisor_delta_cpu=max(0,w.info['supervisor_cpu_seconds']-before)))
  else:finish(w)
 handle=patch.object(n.Native,'finish',observed);handle.start()
 try:
  # 1 MiB per trial, two engines, repeated three times on one nice19 CPU.
  fixture.set_tensors([('data',[0x3f80,0xbf80,0x3eaa,0x3880,0x3300,0x3301,0,0x8000]*65536)])
  for trial in range(3):
   for engine in ('reference','native'):
    row={'trial':trial,'engine':engine,'payload_bytes':1048576}
    for phase in ('convert_internal_verify','independent_verify'):
     count=len(observations);cpu=time.process_time();wall=time.perf_counter()
     if phase.startswith('convert'):candidate.convert(str(fixture.source),str(fixture.parent),engine+str(trial),fixture.anchor,selection=engine)
     else:candidate.verify(str(fixture.source),str(fixture.parent/(engine+str(trial))),fixture.anchor,selection=engine)
     elapsed=time.perf_counter()-wall
     coordinator_cpu=time.process_time()-cpu;owned=observations[count:];combined=coordinator_cpu+sum(x['worker_cpu_seconds']+x['supervisor_delta_cpu'] for x in owned)
     row[phase]={'seconds':elapsed,'MiB_s':1/elapsed,'coordinator_cpu_seconds':coordinator_cpu,'combined_cpu_percent':100*combined/elapsed,'combined_sample_rss_kib':max([x['combined_sample_rss_kib'] for x in owned],default=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    rows.append(row)
  raw=(b'\x80\x3f\x01\x33'*32768);kernel={}
  for i in range(5):
   with candidate.engine('native') as w:
    startups.append(w.startup_seconds);stats=c.empty_stats();t=time.perf_counter()
    for _ in range(8):out=w.block(raw,stats)
    conversion=time.perf_counter()-t;t=time.perf_counter();stats=c.empty_stats()
    for _ in range(8):w.block(raw,stats,out)
    kernel.setdefault('conversion_seconds',[]).append(conversion);kernel.setdefault('verification_seconds',[]).append(time.perf_counter()-t)
  result={'status':'PASS','payload_bytes':1048576,'rows':rows,'startup_seconds':startups,'supervised_block_1MiB':kernel,'coordinator_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'coordinator_cpu_seconds':time.process_time(),'worker_observations':observations,'projection_minutes':{engine:53791996928/1048576*sum(statistics.median(r[p]['seconds'] for r in rows if r['engine']==engine) for p in ('convert_internal_verify','independent_verify'))/60 for engine in ('reference','native')},'notes':'Warm cache; three trials, 1 MiB. Native workers are siblings; worker CPU/RSS from native wait4 usage events. Hashes/staging/fsync included, external receipts and cold production disk excluded.'}
  print(json.dumps(result),flush=True)
 finally:handle.stop();fixture.doCleanups()
