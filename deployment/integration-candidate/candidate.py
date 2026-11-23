"""Synthetic-only integration entry points. No production CLI or implicit fallback."""
from contextlib import contextmanager
from native_engine import ACTIVE,Native
@contextmanager
def engine(selection):
 from lifecycle_support import acquisition_signals
 if selection not in ('native','reference'):raise ValueError('explicit engine required')
 previous=ACTIVE.get()
 if previous is not None:raise RuntimeError('nested native engine')
 worker=None;token=None;failure=None;traceback=None
 try:
  if selection=='native':
   with acquisition_signals():
    # Own the Python supervisor before its initializer can launch a child.
    worker=Native.__new__(Native)
    Native.__init__(worker)
  with acquisition_signals():token=ACTIVE.set(worker)
  yield worker
  if worker:worker.finish()
 except BaseException as error:
  failure=error;traceback=error.__traceback__
  if worker is not None and hasattr(worker,'worker_id'):worker.abort()
 finally:
  restored=False
  for attempt in range(2):
   try:
    if ACTIVE.get() is previous:
     restored=True;break
    if token is not None:
     ACTIVE.reset(token);token=None
    else:ACTIVE.set(previous)
    restored=ACTIVE.get() is previous
    if restored:break
   except BaseException as cleanup:
    if worker is not None and hasattr(worker,'worker_id'):worker.diagnostic('context_reset',cleanup)
    if failure is None:failure=cleanup;traceback=cleanup.__traceback__
  if worker is not None and hasattr(worker,'worker_id'):
   worker.context_pending=not restored
   if failure is not None or not restored:worker.abort()
   worker.release_if_complete()
   if failure is not None:worker.attach(failure)
 if failure is not None:raise failure.with_traceback(traceback)
 if not restored:raise RuntimeError('engine context cleanup incomplete')
def convert(source,parent,name,anchor,*,selection):
 import convert_fp16
 from audit_io import LifecycleError
 state=dict(phase='preflight',promoted=False,durability='not_published')
 try:
  with engine(selection):return convert_fp16.convert(source,parent,name,anchor,reserve_bytes=0,lifecycle=state)
 except LifecycleError:raise
 except BaseException as error:
  from native_engine import record_worker_state
  record_worker_state(state,error)
  raise LifecycleError(state,error) from error
def verify(source,output,anchor,*,selection):
 import verify_fp16
 with engine(selection):return verify_fp16.verify(source,output,anchor)
