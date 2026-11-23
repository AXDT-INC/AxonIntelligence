"""Destructive-to-own-session cases, each under a fresh native subreaper harness.
The outer runner owns fixtures, including when this coordinator exits abruptly.
"""
import json,os,signal,sys
from pathlib import Path
from unittest.mock import patch
import candidate,native_engine as n,convert_fp16 as c,test_conversion as fixtures

def main(case,workspace):
 workspace=Path(workspace)
 # The test runner owns this directory; abrupt exit cannot escape its cleanup.
 import tempfile
 temporary=tempfile.TemporaryDirectory
 def local_temp(*a,**kw):kw['dir']=str(workspace);return temporary(*a,**kw)
 with patch.object(tempfile,'TemporaryDirectory',local_temp):f=fixtures.ConversionTests();f.setUp()
 (workspace/'artifact-path').write_text(str(f.parent/'commit-result'))
 state={};worker=None;checkpoint=[];failure=None
 def evidence(label):
  record={'label':label,'artifact_exists':(f.parent/'commit-result').exists(),'promoted':state.get('promoted'),'durability':state.get('durability')}
  checkpoint.append(record);print(json.dumps({'checkpoint':record}),flush=True)
 needles={
 'supervisor_before_begin':"__import__('native_engine').require_publication_authorized(state)",
 'supervisor_after_grant':"state.update(phase='atomic_promotion'",
 'supervisor_after_rename':"state.update(phase='post_promotion_fsync'",
 'supervisor_after_confirmed':"__import__('native_engine').complete_publication(state)",
 'coordinator_after_grant':"state.update(phase='atomic_promotion'",
 'coordinator_after_rename':"state.update(phase='post_promotion_fsync'",
 'coordinator_after_confirmed':"__import__('native_engine').complete_publication(state)"}
 line=next((i for i,s in enumerate(Path(c.__file__).read_text().splitlines(),1) if case in needles and needles[case] in s),None)
 def trace(frame,event,arg):
  if line is not None and event=='line' and frame.f_code is c.convert.__code__ and frame.f_lineno==line and not checkpoint:
   evidence(case)
   if case.startswith('coordinator'):os._exit(73)
   # Only our supervisor parent, created by this dedicated test harness.
   os.kill(os.getppid(),signal.SIGKILL)
   worker.control(3) # force detection; no retry
  return trace
 exchange=n.SESSION.exchange
 def fault(op,token,payload=b'',timeout=7):
  target=7 if 'begin' in case else 8
  if op!=target or checkpoint:return exchange(op,token,payload,timeout)
  if case.startswith('lost_'):
   response=exchange(op,token,payload,timeout);evidence(case);n.SESSION.close();raise n.WorkerError('discarded response; no retry')
  if case.startswith('partial_'):
   evidence(case);n.SESSION.seq+=1
   frame=n.HEADER.pack(0x41585332,2,n.SESSION.seq,op,token,len(payload))+payload
   os.write(4,frame[:12] if op==7 else frame[:-1]);n.SESSION.close();raise n.WorkerError('partial frame; no retry')
  if case=='duplicate_result_response':
   response=exchange(op,token,payload,timeout);evidence(case)
   # A duplicate frame uses the already consumed sequence, not a new operation.
   os.write(4,n.HEADER.pack(0x41585332,2,n.SESSION.seq,op,token,len(payload))+payload)
   n.SESSION.exchange(3,token,timeout=2)
   raise AssertionError('duplicate frame accepted')
  if case=='invalid_result_response':
   response=json.loads(exchange(op,token,payload,timeout));evidence(case);response['commit_state']='not_published';return json.dumps(response).encode()
  return exchange(op,token,payload,timeout)
 try:
  with candidate.engine('native') as worker:
   with patch.object(n.SESSION,'exchange',fault):
    sys.settrace(trace)
    c.convert(str(f.source),str(f.parent),'commit-result',f.anchor,0,state)
 except BaseException as error:failure={'type':type(error).__name__,'message':str(error)}
 finally:
  sys.settrace(None)
  print(json.dumps({'case':case,'state':state,'failure':failure,'artifact_exists':(f.parent/'commit-result').exists(),'worker':worker.snapshot() if worker else None}),flush=True)
  # Fixture deliberately retained for the outer harness's independent existence check.
  f.temporary._finalizer.detach();n.SESSION.close()
