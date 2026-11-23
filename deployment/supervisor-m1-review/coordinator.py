"""Only invoked by native supervisor. Synthetic tests and benchmarks only."""
import os,sys,json,unittest,time,signal
from pathlib import Path
ROOT=Path(__file__).absolute().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'));sys.dont_write_bytecode=True
import native_engine as n
n.SESSION=n.Session()
def scenario(name):
 s=n.SESSION;s.token+=1;token=s.token
 if name=='normal':
  with __import__('candidate').engine('native') as w:
   assert w.block(b'\x80\x3f',__import__('convert_fp16').empty_stats())==b'\0\x3c'
  return {'status':'PASS','worker':w.snapshot()}
 if name in ('worker_crash','worker_signal','unexpected_response'):
  import candidate,convert_fp16 as c
  from test_conversion import ConversionTests
  from unittest.mock import patch
  from audit_io import LifecycleError
  f=ConversionTests();f.setUp();state={}
  try:
   with candidate.engine('native') as w:
    if name!='unexpected_response':os.kill(w.info['pid'],signal.SIGKILL if name=='worker_crash' else signal.SIGTERM)
    try:c.convert(str(f.source),str(f.parent),'result',f.anchor,0,state)
    except LifecycleError:
     assert state['promoted'] is False and not (f.parent/'result').exists()
     raise n.WorkerError('expected worker failure')
   raise AssertionError('worker failure accepted')
  except n.WorkerError:return {'status':'PASS','lifecycle':state}
  finally:f.doCleanups()
 if name=='partial':
  os.write(4,b'partial');s.close();return {'status':'PASS','launch':'UNKNOWN'}
 if name in ('bad_version','oversized'):
  os.write(4,n.HEADER.pack(0x41585332,99 if name=='bad_version' else 2,1,1,token,131201 if name=='oversized' else 0));s.close();return {'status':'PASS','launch':'UNKNOWN'}
 if name=='supervisor_failure':
  result=json.loads(s.exchange(1,token));os.kill(os.getppid(),signal.SIGKILL)
  try:s.exchange(3,token)
  except n.WorkerError:return {'status':'PASS','launch':'UNKNOWN','last_observed':result}
  raise AssertionError('supervisor failure not detected')
 try:
  ready=json.loads(s.exchange(1,token,timeout=3))
  if name=='partial_after_start':
   os.write(4,b'partial ACK');s.close();return {'status':'PASS','launch':'UNKNOWN','last_observed':ready}
  if name=='before_go':
   # A numerical request without ACK must be rejected by native owner.
   try:s.exchange(5,token,n.HEADER.pack(n.MAGIC,1,1,1,2,80)+n.STATS.pack(*([0]*10))+b'\0\0')
   except n.WorkerError:return {'status':'PASS','launch':'UNKNOWN'}
   raise AssertionError('pre-GO numerical execution accepted')
  if name in ('before_ack','stop_before_ack'):
   time.sleep(2.3);result=json.loads(s.exchange(3,token));assert result['reaped'];return {'status':'PASS','worker':result}
  if name=='lost_ack':
   try:s.exchange(2,token,timeout=3)
   except n.WorkerError:return {'status':'PASS','launch':'UNKNOWN'}
   raise AssertionError('lost ACK was accepted')
  if name=='duplicate':
   try:s.exchange(1,token)
   except n.WorkerError:return {'status':'PASS','launch':'UNKNOWN'}
   raise AssertionError('duplicate START accepted')
  if name in ('kill_failure','wait_failure','after_ack','close_failure','kill_unresolved','wait_unresolved'):
   s.exchange(2,token);first=json.loads(s.exchange(4,token));result=json.loads(s.exchange(4,token))
   if not result['reaped']:result=json.loads(s.exchange(4,token))
   if name.endswith('unresolved'):
    assert not first['reaped'] and first['process_state']=='cleanup_incomplete'
    assert result['error_count']>=first['error_count']
   assert result['reaped']
   if name!='after_ack':assert len(result['cleanup_errors'])>=2
   return {'status':'PASS','first_cleanup':first,'worker':result}
  raise AssertionError('unsupported scenario '+name)
 except (KeyboardInterrupt,n.WorkerError) as e:
  s.close()
  if name not in ('child_before_delivery','descriptor_checkpoint','lost_ready'):raise
  return {'status':'PASS','launch':'UNKNOWN','error_type':type(e).__name__}
def main():
 mode,name=sys.argv[1:]
 if mode=='scenario':print(json.dumps(scenario(name)),flush=True);return 0
 if mode=='benchmark':__import__('benchmark').main();return 0
 suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
 def names(s):
  for t in s:
   if isinstance(t,unittest.TestSuite):yield from names(t)
   else:yield t.id()
 ids=list(names(suite));result=unittest.TextTestRunner(verbosity=2).run(suite)
 print(json.dumps({'status':'PASS' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,'tests':ids,'failures':[(str(t),e) for t,e in result.failures],'errors':[(str(t),e) for t,e in result.errors],'skipped':result.skipped}),flush=True)
 return not result.wasSuccessful()
if __name__=='__main__':
 try:status=main()
 except BaseException as e:
  try:print(json.dumps({'status':'FAIL','error_type':type(e).__name__,'error':str(e),'launch':'UNKNOWN'}),flush=True)
  except BaseException:pass
  status=1
 n.SESSION.close();sys.exit(status)
