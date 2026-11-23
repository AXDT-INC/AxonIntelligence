import unittest,struct,json,hashlib,os,signal
from pathlib import Path
from unittest.mock import patch
import test_conversion as fixtures
import candidate,native_engine as n,convert_fp16 as c
from audit_io import LifecycleError

class NativeFixtures(fixtures.ConversionTests):
 def run_conversion(self,name='fp16'):
  return candidate.convert(str(self.source),str(self.parent),name,self.anchor,selection='native')
 def test_scalar_matrix(self):
  self.set_tensors([('scalar',[0x3f80]),('matrix',[0x3301]*6)])
  fixtures.rewrite_header(self.shard,lambda h:(h['scalar'].update(shape=[]),h['matrix'].update(shape=[2,3])))
  self.anchor=fixtures.source_manifest(self.source);self.run_conversion()
  self.assertEqual(candidate.verify(str(self.source),str(self.parent/'fp16'),self.anchor,selection='native')['status'],'PASS')
 def test_promotion_boundary_unknown(self):
  original=c.promote;state={}
  def interrupt(*a):original(*a);raise KeyboardInterrupt('after rename')
  with candidate.engine('native'),patch.object(c,'promote',interrupt):
   with self.assertRaises(LifecycleError):c.convert(str(self.source),str(self.parent),'fp16',self.anchor,0,state)
  self.assertTrue((self.parent/'fp16').exists());self.assertIsNone(state['promoted']);self.assertEqual(state['durability'],'unknown')
 def test_cleanup_after_confirmed_publication(self):
  original=n.Native.finish;state={};calls=[]
  def interrupt(w):
   original(w);calls.append(w.closed)
   if state.get('promoted'):raise SystemExit('post publication')
  with patch.object(n.Native,'finish',interrupt):
   with self.assertRaises(SystemExit):
    with candidate.engine('native'):c.convert(str(self.source),str(self.parent),'fp16',self.anchor,0,state)
  self.assertTrue(state['promoted']);self.assertTrue((self.parent/'fp16').exists());self.assertEqual(state['durability'],'confirmed');self.assertTrue(all(calls))
 def test_verifier_failure_blocks_publication(self):
  original=n.Native.block;state={}
  def bad(w,raw,stats,actual=None):
   if actual is not None:actual=bytes([actual[0]^1])+actual[1:]
   return original(w,raw,stats,actual)
  with patch.object(n.Native,'block',bad):
   with self.assertRaises(Exception):
    with candidate.engine('native'):c.convert(str(self.source),str(self.parent),'fp16',self.anchor,0,state)
  self.assertFalse(state['promoted']);self.assertFalse((self.parent/'fp16').exists())
 def test_rehashed_provenance_rejected(self):
  self.run_conversion();out=self.parent/'fp16';r=out/'CONVERSION.json';record=json.loads(r.read_text());record['candidate_engine']['supervisor_sha256']='0'*64;r.write_bytes(__import__('audit_io').canonical(record));fixtures.update_output_manifest(out)
  with self.assertRaises(n.WorkerError):candidate.verify(str(self.source),str(out),self.anchor,selection='reference')

class Numerical(unittest.TestCase):
 def test_exhaustive_all_65536(self):
  data=(n.ROOT/'test-workspace'/'exhaustive.bin').read_bytes();self.assertEqual(len(data),65536*83)
  counts={0:0,1:0,2:0}
  for bits in range(65536):
   rec=data[bits*83:(bits+1)*83];stats=c.empty_stats()
   try:expected=bytes(c.reference_convert_block(struct.pack('<H',bits),stats))
   except Exception:self.assertEqual(rec[0],1 if bits&0x7f80==0x7f80 else 2)
   else:self.assertEqual(rec[0],0);self.assertEqual(rec[1:3],expected);self.assertEqual(dict(zip(n.KEYS,n.STATS.unpack(rec[3:]))),stats)
   counts[rec[0]]+=1
  self.assertEqual(counts,{0:36608,1:256,2:28672})
 def test_all_valid_through_ipc(self):
  raw=b''.join(struct.pack('<H',i) for i in range(65536) if (i&0x7fff)<=0x477f)
  stats=c.empty_stats();expected=bytes(c.reference_convert_block(raw,stats))
  with candidate.engine('native') as w:
   outstats=c.empty_stats();out=w.block(raw,outstats);self.assertEqual(out,expected);self.assertEqual(outstats,stats)
   verify=c.empty_stats();w.block(raw,verify,out);self.assertEqual(verify,stats)
 def test_boundaries_and_exact_statistics(self):
  for size in (0,2,65534,65536,65538,131072):
   raw=(b'\x00\x80\x01\x33'*(size//4+1))[:size];ref=c.empty_stats();expected=bytes(c.reference_convert_block(raw,ref))
   with candidate.engine('native') as w:
    got=c.empty_stats();self.assertEqual(w.block(raw,got),expected);self.assertEqual(got,ref)
 def test_nonfinite_overflow(self):
  for bits in (0x7f80,0x7fc0,0xff80,0xffff,0x4780,0xc780):
   with self.assertRaises(n.WorkerError),candidate.engine('native') as w:w.block(struct.pack('<H',bits),c.empty_stats())
 def test_corruption_rejected(self):
  for bits in (0,0x8000,0x3300,0x3301,0x3f80,0x477f):
   raw=struct.pack('<H',bits);out=bytes(c.reference_convert_block(raw,c.empty_stats()))
   with self.assertRaises(n.WorkerError),candidate.engine('native') as w:w.block(raw,c.empty_stats(),bytes([out[0]^1,out[1]]))
 def test_primary_systemexit_preserved(self):
  with self.assertRaisesRegex(SystemExit,'primary'):
   with candidate.engine('native') as w:raise SystemExit('primary')
  self.assertTrue(w.closed);self.assertTrue(w.snapshot()['reaped'])
 def test_primary_keyboardinterrupt_preserved(self):
  with self.assertRaises(KeyboardInterrupt):
   with candidate.engine('native') as w:raise KeyboardInterrupt()
  self.assertTrue(w.closed)
 def test_repeated_abort_preserves_state(self):
  w=n.Native();a=w.abort();b=w.abort();self.assertEqual(a,b);self.assertTrue(a['reaped'])
 def test_invalid_block_rejected_before_processing(self):
  with candidate.engine('native') as w:
   for bad in (b'x',b'x'*131074,bytearray(2)):
    with self.assertRaises(n.WorkerError):w.block(bad,c.empty_stats())
