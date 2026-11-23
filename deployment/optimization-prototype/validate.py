import os,sys,struct,time,ctypes,random
from pathlib import Path
from prototype import Engine,Statistics,Rejected,BLOCK,canonical,ROOT
sys.path.insert(0,str(ROOT.parent))
import convert_fp16 as reference
import verify_fp16 as oracle
from audit_io import AuditError
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
if os.nice(0)<19:os.nice(19-os.nice(0))
start=time.monotonic();engines=[Engine(),Engine(True)];counts=dict(valid=0,nonfinite=0,overflow=0,bit_flip_rejections=0)
def reject(call,code):
    try:call()
    except Rejected as e:assert e.code==code,(e.code,code)
    else:raise AssertionError('accepted invalid input')
valid=[]
for b in range(65536):
    raw=struct.pack('<H',b);st=reference.empty_stats()
    try:out=bytes(reference.convert_block(raw,st))
    except AuditError:
        code=1 if b&0x7f80==0x7f80 else 2
        counts['nonfinite' if code==1 else 'overflow']+=1
        for e in engines:
            s=Statistics();reject(lambda:e.convert_block(raw,s),code)
            reject(lambda:e.verify_block(raw,b'\0\0',s),code)
            assert s.record()==reference.empty_stats()
    else:
        counts['valid']+=1;valid.append(b)
        assert struct.unpack('<H',out)[0]==oracle.expected_half_bits(b)
        v=oracle.initial_statistics();oracle.verify_blocks(raw,out,v);assert v==st
        for e in engines:
            s=Statistics();assert e.convert_block(raw,s)==out and s.record()==st
            s=Statistics();e.verify_block(raw,out,s);assert s.record()==st
        for bit in range(16):
            bad=struct.pack('<H',int.from_bytes(out,'little')^(1<<bit))
            reject(lambda:engines[0].verify_block(raw,bad,Statistics()),4)
            counts['bit_flip_rejections']+=1
    if b%4096==0:time.sleep(.02)
e=engines[0];cases=[]
rng=random.Random(123)
raw=b''.join(struct.pack('<H',b) for b in valid+list(reversed(valid))+rng.choices(valid,k=4096))
s=reference.empty_stats();expected=bytes(reference.convert_block(raw,s))
for size in (2,14,126,65534,65536):
    cs=Statistics();vs=Statistics();parts=[]
    for i in range(0,len(raw),size):
        block=raw[i:i+size];out=e.convert_block(block,cs);parts.append(out);e.verify_block(block,out,vs)
    assert b''.join(parts)==expected and cs.record()==vs.record()==s
cases.append('ordered_statistics_and_block_boundaries')
for engine in engines:
    for bad in (b'x',b'\0'*(BLOCK+2),bytearray(2)):
        reject(lambda:engine.convert_block(bad,Statistics()),3)
    reject(lambda:engine.verify_block(b'\0\0',b'',Statistics()),3)
    for bad,code in ((0x7f80,1),(0x4780,2)):
        stats=Statistics();reject(lambda:engine.convert_block(b'\0\0'+struct.pack('<H',bad),stats),code)
        assert stats.record()==reference.empty_stats()
    assert engine.convert_block(b'',Statistics())==b''
cases.append('malformed_empty_oversized_and_failure_atomic_statistics')
for shape,data in (([],b'\x80\x3f'),([2,3],b'\x80\x3f'*6),([2,0],b'')):
    a=e.tensor('synthetic',shape,[data]);b=e.tensor('synthetic',shape,[data[:2],data[2:]])
    assert canonical(a)==canonical(b)
for shape,data in (([-1],b''),([True],b''),([1],b''),([0],b'\0\0'),([100000000],b'')):
    try:e.tensor('invalid',shape,[data])
    except ValueError:pass
    else:raise AssertionError('shape/payload accepted')
cases.append('scalar_multidimensional_empty_truncated_trailing_and_deterministic_receipts')
for exc in (KeyboardInterrupt,SystemExit,OSError):
    def interrupted():
        yield b'\x80\x3f'
        raise exc('injected')
    try:e.tensor('interrupted',[2],interrupted())
    except exc:pass
    else:raise AssertionError('interruption swallowed')
cases.append('interrupted_processing_no_success_receipt')
libm=ctypes.CDLL('libm.so.6');prior=libm.fegetround()
try:
    for mode in (0x400,0x800,0xc00):
        assert libm.fesetround(mode)==0
        reject(lambda:e.convert_block(b'\0\0',Statistics()),5)
        reject(lambda:e.verify_block(b'\0\0',b'\0\0',Statistics()),5)
finally:assert libm.fesetround(prior)==0
cases.append('non_nearest_floating_environment_rejected')
result=dict(status='PASS_SYNTHETIC_ONLY',encodings=65536,counts=counts,regression_groups=cases,statistics='exact V3 equality including ordered binary64 sums',ubsan='exhaustive valid and invalid encodings passed',elapsed_seconds=time.monotonic()-start,production_model_verified=False)
with (ROOT/'validation-results.json').open('xb') as f:f.write(canonical(result))
print(canonical(result).decode())
