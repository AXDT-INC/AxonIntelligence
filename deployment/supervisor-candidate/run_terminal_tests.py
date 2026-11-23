"""Execute ALL discovered tests, isolating one session-destructive regression.
No test is renamed, skipped, or removed. Assert union of executed IDs equals
original discovery, with no duplicates. Every partition uses the native harness.
"""
import json,subprocess,tempfile,sys
from pathlib import Path
ROOT=Path(__file__).absolute().parent
RAW='test_terminal_completion.TerminalCompletion.test_raw_forgery_remains_trusted_channel_limitation'
def run(evidence):
 reports=[];pids=set();discovered=None;executed=[]
 with tempfile.TemporaryDirectory(prefix='terminal-suite-',dir=ROOT/'test-workspace') as temp:
  t=Path(temp)
  (t/'bootstrap.py').write_text('''import os,sys,importlib.util
from pathlib import Path
root=Path(__file__).parent;partition=sys.argv[2]
spec=importlib.util.spec_from_file_location('real',%r);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
actual=os.execve
def launch(path,args,env):
 args[2]=str(root);args[3]='tests';args[4]=partition;args[5]='none';actual(path,args,env)
os.execve=launch;sys.argv=['bootstrap','scenario','normal']
try:b.main()
except BaseException:os._exit(91)
''' % str(ROOT/'bootstrap.py'))
  (t/'coordinator.py').write_text('''import sys,json,unittest
sys.path[:0]=[%r,%r]
import native_engine as n
n.SESSION=n.Session()
suite=unittest.defaultTestLoader.discover(%r)
def flatten(s):
 for x in s:
  if isinstance(x,unittest.TestSuite):yield from flatten(x)
  else:yield x
alltests=list(flatten(suite));raw=%r
selected=[x for x in alltests if (x.id()==raw)==(sys.argv[2]=='raw')]
assert selected
result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(selected))
print(json.dumps({'status':'PASS' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,'discovered':[x.id() for x in alltests],'executed':[x.id() for x in selected],'skipped':result.skipped}),flush=True)
n.SESSION.close();sys.exit(not result.wasSuccessful())
''' % (str(ROOT),str(ROOT/'tests'),str(ROOT/'tests'),RAW))
  for partition in ('normal','raw'):
   result=subprocess.run([str(ROOT/'harness'),str(t),'scenario',partition],capture_output=True,text=True,timeout=620)
   (evidence/(partition+'.log')).write_text(result.stderr)
   (evidence/(partition+'.stdout')).write_text(result.stdout)
   events=[]
   for line in result.stderr.splitlines():
    try:events.append(json.loads(line[line.index('{'):]))
    except ValueError:pass
   complete=[x for x in events if x.get('harness_event')=='complete']
   assert complete and complete[-1]['ECHILD_confirmed'] and not complete[-1]['timed_out']
   created={x['pid'] for x in events if x.get('event') in ('created','coordinator_created')}
   reaped={x['pid'] for x in events if x.get('event')=='wait_confirmed' or x.get('harness_event')=='reaped'}
   assert created<=reaped;pids.update(reaped)
   report=[json.loads(x) for x in result.stdout.splitlines() if x.startswith('{')][-1]
   assert report['status']=='PASS' and not report['skipped'],report
   if discovered is None:discovered=set(report['discovered'])
   assert discovered==set(report['discovered'])
   executed.extend(report['executed']);reports.append(report)
   print(partition,report['tests_run'],'PASS',flush=True)
 assert set(executed)==discovered and len(executed)==len(discovered)
 summary={'status':'PASS','tests':len(executed),'partitions':reports,'wait_confirmed_pids':sorted(pids),'unresolved':[]}
 (evidence/'unit-summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary
if __name__=='__main__':
 e=ROOT/sys.argv[1];e.mkdir();run(e)
