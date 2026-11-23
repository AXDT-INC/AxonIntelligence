import os,sys,time,struct,hashlib,resource,statistics
from prototype import ROOT,Engine,Statistics,BLOCK,canonical
sys.path.insert(0,str(ROOT.parent))
import convert_fp16 as ref
import verify_fp16 as ver
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
if os.nice(0)<19:os.nice(19-os.nice(0))
def memory():
    return {line.split(':')[0]:line.split(':',1)[1].strip() for line in Path('/proc/self/status').read_text().splitlines() if line.startswith(('VmRSS:', 'VmHWM:'))}
from pathlib import Path
baseline_memory=memory()
e=Engine();size=2*1024**2
seed=b''.join(struct.pack('<H',b) for b in range(65536) if (b&0x7fff)<=0x477f)
raw=(seed*((size+len(seed)-1)//len(seed)))[:size]
blocks=[raw[i:i+BLOCK] for i in range(0,size,BLOCK)]
rows=[];expected=None;expected_stats=None
for implementation in ('V3','prototype'):
    for trial in range(3):
        row={'implementation':implementation,'trial':trial+1,'payload_bytes':size}
        for phase in ('conversion','verification'):
            st=ref.empty_stats() if implementation=='V3' else Statistics()
            outputs=[];cpu=time.process_time();wall=time.perf_counter()
            for i,block in enumerate(blocks):
                if phase=='conversion':
                    outputs.append(bytes(ref.convert_block(block,st)) if implementation=='V3' else e.convert_block(block,st))
                elif implementation=='V3':ver.verify_blocks(block,expected[i],st)
                else:e.verify_block(block,expected[i],st)
            elapsed=time.perf_counter()-wall;cpu=time.process_time()-cpu
            stats=st if implementation=='V3' else st.record()
            row[phase]={'wall_seconds':elapsed,'cpu_seconds':cpu,'cpu_percent_one_core':100*cpu/elapsed,'MiB_per_second':size/1048576/elapsed}
            if expected is None:expected=outputs;expected_stats=stats
            assert stats==expected_stats
            if phase=='conversion':assert outputs==expected
        row['proc_memory']=memory()
        row['peak_RSS_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rows.append(row);time.sleep(.2)
summary={}
for implementation in ('V3','prototype'):
    subset=[r for r in rows if r['implementation']==implementation]
    summary[implementation]={p:{'median_MiB_per_second':statistics.median(r[p]['MiB_per_second'] for r in subset),'min_MiB_per_second':min(r[p]['MiB_per_second'] for r in subset),'max_MiB_per_second':max(r[p]['MiB_per_second'] for r in subset)} for p in ('conversion','verification')}
    c=summary[implementation]['conversion']['median_MiB_per_second'];v=summary[implementation]['verification']['median_MiB_per_second']
    summary[implementation]['estimated_three_pass_hours']=53791996928/1048576*(1/c+2/v)/3600
result={'schema':'synthetic-benchmark.v2','baseline_memory':baseline_memory,'payload_sha256':hashlib.sha256(raw).hexdigest(),'output_sha256':hashlib.sha256(b''.join(expected)).hexdigest(),'statistics':expected_stats,'output_and_statistics_equivalent':True,'trials':rows,'summary':summary,'three_pass_speedup':summary['V3']['estimated_three_pass_hours']/summary['prototype']['estimated_three_pass_hours'],'affinity':sorted(os.sched_getaffinity(0)),'nice':os.nice(0),'notes':'2 MiB/trial, three trials each. Timed block conversion/verification and output copying; excludes disk, hashing, manifest and lifecycle operations. Process RSS is cumulative high-water, includes fixtures. Uniform valid BF16 encoding distribution; not model data.'}
with (ROOT/'benchmark-results-v2.json').open('xb') as f:f.write(canonical(result))
print(canonical(result).decode())
