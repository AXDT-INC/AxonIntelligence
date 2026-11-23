"""Own-session crash/IPC regression driver. No production paths or external services.
Every case executes beneath the unchanged native subreaper; its fixture root is
an exclusive TemporaryDirectory below test-workspace and is removed after reap.
"""
import json,os,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).absolute().parent
CASES=['supervisor_before_begin','supervisor_after_grant','supervisor_after_rename','supervisor_after_confirmed','coordinator_after_grant','coordinator_after_rename','coordinator_after_confirmed','lost_begin_response','lost_result_response','partial_begin_frame','partial_result_frame','duplicate_result_response','invalid_result_response']
def run(case,evidence):
 with tempfile.TemporaryDirectory(prefix='commit-process-',dir=ROOT/'test-workspace') as temp:
  t=Path(temp)
  (t/'bootstrap.py').write_text('''import os,sys,importlib.util
from pathlib import Path
root=Path(__file__).parent
spec=importlib.util.spec_from_file_location('authenticated_bootstrap',%r)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
real=os.execve
case=sys.argv[2]
def launch(path,args,env):
 args[2]=str(root);args[3]='commit-test';args[4]=case;args[5]='none'
 real(path,args,env)
os.execve=launch;sys.argv=['bootstrap','scenario','normal']
try:b.main()
except BaseException:os._exit(91)
''' % str(ROOT/'bootstrap.py'))
  (t/'coordinator.py').write_text('''import sys
from pathlib import Path
sys.path[:0]=[%r,%r]
import native_engine as n
n.SESSION=n.Session()
import commit_process_cases
commit_process_cases.main(sys.argv[2],Path(__file__).parent)
''' % (str(ROOT),str(ROOT/'tests')))
  result=subprocess.run([str(ROOT/'harness'),str(t),'scenario',case],capture_output=True,text=True,timeout=620)
  (evidence/(case+'.log')).write_text(result.stderr)
  (evidence/(case+'.stdout')).write_text(result.stdout)
  events=[]
  for line in result.stderr.splitlines():
   try:events.append(json.loads(line[line.index('{'):]))
   except ValueError:pass
  accounting=[e for e in events if e.get('harness_event')=='complete']
  assert accounting and accounting[-1]['ECHILD_confirmed'] and not accounting[-1]['timed_out'],result.stderr
  created={e['pid'] for e in events if e.get('event') in ('created','coordinator_created')}
  reaped={e['pid'] for e in events if e.get('event')=='wait_confirmed' or e.get('harness_event')=='reaped'}
  assert created<=reaped,(case,created-reaped)
  for pid in reaped:assert not Path('/proc',str(pid)).exists(),pid
  records=[json.loads(x) for x in result.stdout.splitlines() if x.startswith('{')]
  artifact=Path((t/'artifact-path').read_text());assert artifact.is_relative_to(t)
  exists=artifact.exists()
  expected=case not in ('supervisor_before_begin','supervisor_after_grant','coordinator_after_grant','lost_begin_response','partial_begin_frame')
  assert exists==expected,(case,exists,expected,records)
  end=[e['publication'] for e in events if e.get('event')=='commit_session_end']
  if case.startswith('coordinator') or case in ('lost_begin_response','partial_result_frame'):
   assert end and end[-1]['commit_state']=='UNKNOWN',end
   assert end[-1]['cancellation_complete'] is False
  if case in ('lost_result_response','duplicate_result_response','invalid_result_response'):
   assert end and end[-1]['commit_state']=='published',end
  final=[x for x in records if 'failure' in x]
  if not case.startswith('coordinator'):
   assert final and final[-1]['failure'] is not None,records
   state=final[-1]['state']
   if case=='supervisor_after_rename':assert state['promoted'] is None
   elif expected:assert state['promoted'] is True
   else:assert state['promoted'] is False
  report={'status':'PASS','case':case,'artifact_exists_independently_observed':exists,'wait_confirmed_pids':sorted(reaped),'unresolved':[],'accounting':accounting[-1],'supervisor_final':end,'coordinator_reports':records}
  (evidence/(case+'.json')).write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
  print(case,'PASS',flush=True);return report

def main():
 import sys
 label=sys.argv[1] if len(sys.argv)>1 else 'publication-gate-v3-process'
 assert label.replace('-','').isalnum()
 evidence=ROOT/label;evidence.mkdir()
 reports=[run(case,evidence) for case in CASES]
 summary={'status':'PASS','cases':len(reports),'wait_confirmed_processes':len({p for r in reports for p in r['wait_confirmed_pids']}),'unresolved':[]}
 (evidence/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
