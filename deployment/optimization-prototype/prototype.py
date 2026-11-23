"""Synthetic-only, bounded bytes API. No filesystem/model loading or publication."""
import ctypes
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).absolute().parent
BLOCK=65536
TRIAL_LIMIT=16*1024**2
FIELDS=('elements','changed','zeroed_nonzero','output_subnormal','input_nonfinite','overflow','output_nonfinite')
FLOATS=('max_absolute_error','max_relative_error','sum_absolute_error')

class Statistics(ctypes.Structure):
    _fields_=[(n,ctypes.c_uint64) for n in FIELDS]+[(n,ctypes.c_double) for n in FLOATS]
    def record(self):return {n:getattr(self,n) for n in FIELDS+FLOATS}

class Rejected(ValueError):
    def __init__(self,code):
        self.code=code
        super().__init__({1:'nonfinite',2:'overflow',3:'length',4:'output bit mismatch',5:'rounding mode',6:'state'}[code])

class Engine:
    def __init__(self,sanitized=False):
        name='libprototype-ubsan.so' if sanitized else 'libprototype.so'
        self.library=ctypes.CDLL(str(ROOT/name))
        self.convert=self.library.prototype_convert
        self.verify=self.library.prototype_verify
        for func in (self.convert,self.verify):
            func.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(Statistics)]
            func.restype=ctypes.c_int
        if ctypes.sizeof(Statistics)!=80:raise RuntimeError('unsupported Statistics ABI')

    @staticmethod
    def validate(raw,stats):
        if type(raw) is not bytes or len(raw)>BLOCK or len(raw)%2:raise Rejected(3)
        if not isinstance(stats,Statistics):raise Rejected(6)
        if any(getattr(stats,k)>stats.elements for k in FIELDS[1:]):raise Rejected(6)
        if any(not math.isfinite(getattr(stats,k)) or getattr(stats,k)<0 for k in FLOATS):raise Rejected(6)

    def convert_block(self,raw,stats):
        self.validate(raw,stats)
        src=ctypes.create_string_buffer(raw,max(1,len(raw)))
        dst=ctypes.create_string_buffer(max(1,len(raw)))
        code=self.convert(src,len(raw),dst,len(raw),ctypes.byref(stats))
        if code:raise Rejected(code)
        return dst.raw[:len(raw)]

    def verify_block(self,raw,actual,stats):
        self.validate(raw,stats)
        if type(actual) is not bytes or len(actual)!=len(raw):raise Rejected(3)
        src=ctypes.create_string_buffer(raw,max(1,len(raw)))
        dst=ctypes.create_string_buffer(actual,max(1,len(actual)))
        code=self.verify(src,len(raw),dst,len(actual),ctypes.byref(stats))
        if code:raise Rejected(code)

    def tensor(self,name,shape,blocks,dtype='BF16'):
        """Stream synthetic tensor, return receipt only; never materialize a model.

        An interrupted/failed generator returns no receipt. Any partial results
        remain local memory. No staging or production artifact API exists.
        """
        if not isinstance(name,str) or not name or len(name)>512:raise ValueError('name')
        if not isinstance(shape,list) or len(shape)>16 or any(type(x) is not int or x<0 or x>TRIAL_LIMIT for x in shape):raise ValueError('shape')
        count=math.prod(shape)
        if count*2>TRIAL_LIMIT or dtype!='BF16':raise ValueError('synthetic size/dtype bound')
        seen=0;stats=Statistics();verified=Statistics();sha=hashlib.sha256()
        for raw in blocks:
            if type(raw) is not bytes:raise ValueError('bytes required')
            seen+=len(raw)
            if seen>count*2:raise ValueError('trailing payload')
            out=self.convert_block(raw,stats)
            self.verify_block(raw,out,verified)
            sha.update(out)
        if seen!=count*2:raise ValueError('truncated payload')
        if stats.record()!=verified.record():raise ValueError('statistics mismatch')
        return {'schema':'axonai.synthetic-c-prototype.v1','status':'PASS_SYNTHETIC_ONLY',
                'name':name,'shape':shape,'dtype':'F16','output_sha256':sha.hexdigest(),
                'statistics':stats.record(),'production_model_verified':False}

def canonical(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
