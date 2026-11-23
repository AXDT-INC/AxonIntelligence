"""Detached main supervisor: durable events, heartbeat and bounded process-group cleanup."""
import json,os,signal,subprocess,time
from datetime import datetime,timezone
from pathlib import Path

def append(path,value):
    with Path(path).open('a') as f:
        f.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),**value},sort_keys=True)+'\n')
        f.flush();os.fsync(f.fileno())

def group_alive(pgid):
    try:os.killpg(pgid,0);return True
    except ProcessLookupError:return False

def terminate_group(proc):
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:return
    end=time.monotonic()+10
    while time.monotonic()<end and group_alive(proc.pid):
        proc.poll();time.sleep(0.1)
    if group_alive(proc.pid):
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:raise RuntimeError('torchrun parent survived cleanup')

def supervise(cmd,env,output,runtime_limit=86400):
    run=Path(output);events=run/'supervisor.jsonl';deadline=time.monotonic()+runtime_limit
    append(events,{'event':'supervisor_started','pid':os.getpid(),'ppid':os.getppid(),'session':os.environ.get('TMUX'),'runtime_limit_seconds':runtime_limit,'cmd':cmd})
    with (run/'torchrun.log').open('ab',buffering=0) as log:
        proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
        append(events,{'event':'torchrun_started','pid':proc.pid,'pgid':proc.pid})
        stopping=False;terminated=False
        def request_stop(*_):
            (run/'STOP_REQUEST').touch(exist_ok=True)
        old={s:signal.signal(s,request_stop) for s in [signal.SIGINT,signal.SIGTERM]}
        try:
            while True:
                rc=proc.poll()
                if rc is not None:
                    if rc!=0:raise RuntimeError('torchrun exited '+str(rc))
                    result=json.loads((run/'TRAINING_RESULT.json').read_text())
                    if result.get('success') is not True or result.get('global_step')!=668 or result.get('stop_requested') is not False:raise RuntimeError('main run ended without 668-step completion')
                    append(events,{'event':'completed','returncode':rc,'global_step':668});return result
                left=deadline-time.monotonic()
                if left<=0:
                    os.killpg(proc.pid,signal.SIGKILL)
                    raise TimeoutError('86400-second main runtime deadline')
                if left<=120 and not stopping:
                    request_stop();stopping=True;append(events,{'event':'deadline_checkpoint_stop_requested'})
                if left<=15 and not terminated:
                    os.killpg(proc.pid,signal.SIGTERM);terminated=True
                append(events,{'event':'heartbeat','pid':os.getpid(),'torchrun_pid':proc.pid,'seconds_remaining':round(left,2)})
                os.fsync(log.fileno());time.sleep(min(10,max(0.01,left)))
        except BaseException as exc:
            terminate_group(proc)
            append(events,{'event':'failed','error':repr(exc),'returncode':proc.poll(),'group_alive_after_cleanup':group_alive(proc.pid)})
            raise
        finally:
            for sig,handler in old.items():signal.signal(sig,handler)
            if group_alive(proc.pid):terminate_group(proc)
