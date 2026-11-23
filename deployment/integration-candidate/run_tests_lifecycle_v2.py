"""Complete synthetic suite with independent tracking/reaping of real children."""
import os,sys,unittest,json,subprocess
from pathlib import Path
from unittest.mock import patch
sys.dont_write_bytecode=True
ROOT=Path(__file__).absolute().parent
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
if os.nice(0)<19:os.nice(19-os.nice(0))
import native_engine
suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
def names(s):
 for t in s:
  if isinstance(t,unittest.TestSuite):yield from names(t)
  else:yield t.id()
ids=list(names(suite));children=[];original=subprocess.Popen.__init__
def tracked(proc,*args,**kwargs):
 children.append(proc)
 return original(proc,*args,**kwargs)
with patch.object(subprocess.Popen,'__init__',tracked):
 result=unittest.TextTestRunner(verbosity=2).run(suite)
leaked=[];records=[]
for proc in children:
 if getattr(proc,'pid',None) is None:continue
 if proc.poll() is None:
  leaked.append(proc.pid);proc.kill();proc.wait(timeout=2)
 records.append({'pid':proc.pid,'returncode':proc.returncode,'reaped':proc.poll() is not None})
 for stream in (proc.stdin,proc.stdout,proc.stderr):
  if stream is not None:stream.close()
registry=list(native_engine.OWNED)
receipt={'status':'PASS' if result.wasSuccessful() and not leaked and not registry else 'FAIL','tests_run':result.testsRun,'tests':ids,'failures':[(str(t),e) for t,e in result.failures],'errors':[(str(t),e) for t,e in result.errors],'skipped':result.skipped,'synthetic_only':True,'child_processes':records,'unexpected_live_children_after_tests':leaked,'unresolved_worker_registry':registry,'all_tracked_children_reaped':all(x['reaped'] for x in records)}
with (ROOT/'test-results-worker-lifecycle-v2.json').open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps({k:receipt[k] for k in ('status','tests_run','unexpected_live_children_after_tests','unresolved_worker_registry','all_tracked_children_reaped')}))
sys.exit(receipt['status']!='PASS')
