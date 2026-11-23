"""Full CPU-only candidate suite, no bytecode, one low-priority CPU."""
import os,sys,unittest,json
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).absolute().parent
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
if os.nice(0)<19:os.nice(19-os.nice(0))
suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
def names(s):
 for test in s:
  if isinstance(test,unittest.TestSuite):yield from names(test)
  else:yield test.id()
ids=list(names(suite));result=unittest.TextTestRunner(verbosity=2).run(suite)
receipt={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,'tests':ids,'failures':[(str(t),e) for t,e in result.failures],'errors':[(str(t),e) for t,e in result.errors],'skipped':result.skipped,'synthetic_only':True}
with (ROOT/'test-results.json').open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n')
sys.exit(not result.wasSuccessful())
