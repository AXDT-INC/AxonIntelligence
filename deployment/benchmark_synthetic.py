"""Bounded 2 MiB synthetic benchmark; no production model access or GPU imports."""
import json
import resource
import time
from pathlib import Path
from convert_fp16 import convert_block, empty_stats
from verify_fp16 import verify_blocks, initial_statistics
from audit_io import CHUNK, canonical

SIZE=2*1024**2
# Mixed signs, zeros, normal values, subnormal outputs and underflow; finite only.
pattern=bytes.fromhex('00000080803f80bf80330033013301000180aa3e7f47')
raw=(pattern*(SIZE//len(pattern)+1))[:SIZE]

def measure(fn):
    start=time.perf_counter();cpu=time.process_time()
    value=fn()
    elapsed=time.perf_counter()-start;used=time.process_time()-cpu
    return value,dict(seconds=elapsed,cpu_seconds=used,cpu_percent_one_core=100*used/elapsed,
                      mib_per_second=SIZE/2**20/elapsed,
                      projected_payload_hours=53791996928/SIZE*elapsed/3600)

def convert():
    stats=empty_stats();out=bytearray()
    for i in range(0,SIZE,CHUNK):out.extend(convert_block(raw[i:i+CHUNK],stats))
    return out

output,conversion=measure(convert)
def verify():
    stats=initial_statistics()
    for i in range(0,SIZE,CHUNK):verify_blocks(raw[i:i+CHUNK],output[i:i+CHUNK],stats)
_,verification=measure(verify)
result={'status':'SYNTHETIC_ONLY','bytes':SIZE,'conversion':conversion,'verification':verification,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'projected_conversion_plus_internal_verifier_hours':conversion['projected_payload_hours']+verification['projected_payload_hours'],
        'projected_including_fresh_verifier_hours':conversion['projected_payload_hours']+2*verification['projected_payload_hours'],
        'limitations':'Single 2 MiB in-memory synthetic run, one core. Excludes hashing/disk I/O, filesystem contention, metadata and production distribution. Not a completion-time guarantee.'}
with (Path(__file__).parent/'benchmark-v2.json').open('xb') as f:f.write(canonical(result))
print(json.dumps(result,indent=2))
