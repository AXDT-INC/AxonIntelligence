"""Synthetic-only entrypoints; native engine requires exec-bootstrap session."""
from contextlib import contextmanager
from native_engine import ACTIVE,Native,record_worker_state
@contextmanager
def engine(selection):
 if selection not in ('reference','native'):raise ValueError('explicit engine required')
 if ACTIVE.get() is not None:raise ValueError('nested engine')
 worker=None;token=None;failure=None;tb=None
 try:
  if selection=='native':
   worker=Native.__new__(Native);Native.__init__(worker)
  token=ACTIVE.set(worker)
  yield worker
  if worker:worker.finish()
 except BaseException as e:
  failure=e;tb=e.__traceback__
 finally:
  if worker is not None and hasattr(worker,'info') and failure is not None:worker.abort()
  for attempt in range(2):
   try:
    if ACTIVE.get() is None:break
    if token is not None:ACTIVE.reset(token);token=None
    else:ACTIVE.set(None)
   except BaseException as e:
    if worker is not None and hasattr(worker,'info'):worker.diagnostic('context_reset',e)
    if failure is None:failure=e;tb=e.__traceback__
  if worker is not None and hasattr(worker,'info'):
   worker.context_pending=ACTIVE.get() is not None
   if failure is not None or worker.context_pending:worker.abort()
   if failure is not None:worker.attach(failure)
 if failure is not None:raise failure.with_traceback(tb)
 if ACTIVE.get() is not None:raise RuntimeError('context cleanup incomplete')
def convert(source,parent,name,anchor,*,selection):
 import convert_fp16
 from audit_io import LifecycleError
 state=dict(phase='preflight',promoted=False,durability='not_published')
 try:
  with engine(selection):return convert_fp16.convert(source,parent,name,anchor,reserve_bytes=0,lifecycle=state)
 except LifecycleError:raise
 except BaseException as error:
  record_worker_state(state,error);raise LifecycleError(state,error) from error
def verify(source,output,anchor,*,selection):
 import verify_fp16
 with engine(selection):return verify_fp16.verify(source,output,anchor)
