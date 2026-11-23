"""Offline development driver; all actual child accounting is native harness-owned.
Creates exclusive evidence folder; removes the temporary exhaustive fixture.
Do not interrupt the driver: harness has its own deadline and cleanup watchdog.
"""
import os,sys,json,subprocess,time,hashlib,tempfile
from pathlib import Path
ROOT=Path(__file__).absolute().parent
SCENARIOS=['normal','before_ack','after_ack','partial','partial_after_start','lost_ready','lost_ack','stop_before_ack','child_before_delivery','descriptor_checkpoint','kill_failure','wait_failure','kill_unresolved','wait_unresolved','close_failure','supervisor_failure','bad_version','oversized','duplicate','before_go','worker_crash','worker_signal','unexpected_response']
def run(mode,scenario,evidence,output=None):
 start=time.monotonic()
 with (evidence/(mode+'-'+scenario+'.log')).open('xb') as err:
  result=subprocess.run([str(ROOT/'harness'),str(ROOT),mode,scenario],stdout=output or subprocess.PIPE,stderr=err,timeout=620)
 events=[]
 for line in (evidence/(mode+'-'+scenario+'.log')).read_text().splitlines():
  # unittest prints its prefix on the same line as native stderr events.
  if '{' in line:
   try:events.append(json.loads(line[line.index('{'):]))
   except ValueError:pass
 accounting=[e for e in events if e.get('harness_event')=='complete']
 assert accounting and accounting[-1]['ECHILD_confirmed'] and not accounting[-1]['timed_out']
 created={e['pid'] for e in events if e.get('event') in ('created','coordinator_created')}
 reaped={e['pid'] for e in events if e.get('event')=='wait_confirmed' or e.get('harness_event')=='reaped'}
 assert created<=reaped,('unresolved native child',created-reaped)
 reports=[] if output else [json.loads(x) for x in result.stdout.decode().splitlines() if x.startswith('{')]
 if mode!='exhaustive':assert reports and reports[-1]['status']=='PASS',(mode,scenario,reports)
 if mode in ('tests','benchmark','exhaustive','kernel'):assert result.returncode==0
 record={'mode':mode,'scenario':scenario,'returncode':result.returncode,'seconds':time.monotonic()-start,'reports':reports,'accounting':accounting[-1],'created_pids':sorted(created),'wait_confirmed_pids':sorted(reaped),'unresolved':[]}
 (evidence/(mode+'-'+scenario+'.json')).write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
 print(mode,scenario,'PASS',flush=True)
 return record

def main():
 os.umask(0o022)
 label=sys.argv[1] if len(sys.argv)>1 else 'validation'
 if not label.replace('-','').isalnum():raise ValueError('simple evidence directory name required')
 evidence=ROOT/label;evidence.mkdir(mode=0o755)
 records=[];fixture=ROOT/'test-workspace'/'exhaustive.bin'
 try:
  with fixture.open('xb') as f:records.append(run('exhaustive','none',evidence,f))
  records.append(run('tests','none',evidence))
  for name in SCENARIOS:records.append(run('scenario',name,evidence))
  records.append(run('benchmark','none',evidence));records.append(run('kernel','none',evidence))
 finally:
  if fixture.exists():fixture.unlink()
 report={'status':'PASS','runs':len(records),'tests':records[1]['reports'][-1]['tests_run'],'scenarios':len(SCENARIOS),'unique_wait_confirmed_processes':len({p for r in records for p in r['wait_confirmed_pids']}),'unresolved':[],'records':[r['mode']+'-'+r['scenario'] for r in records]}
 (evidence/'summary.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
