import io,json,sys,unittest
from unittest.mock import patch
import convert_fp16 as c

class CleanupTests(unittest.TestCase):
    def run_case(self,published,cleanup=None,work_error=None,report_error=None):
        observed={'published':False}
        class Destination:
            def __init__(self,path):pass
            def __enter__(self):return self
            def publish(self,result):pass
            def __exit__(self,*args):
                if cleanup:raise cleanup
        def work(*args,lifecycle):
            observed['published']=published
            lifecycle.update(phase='published' if published else 'pre_publication',promoted=published,durability='confirmed' if published else 'not_published')
            if work_error:raise work_error
            return dict(status='PASS',**lifecycle)
        out,err=io.StringIO(),io.StringIO()
        original=c.emit_receipt
        def report(value):
            if report_error:raise report_error
            original(value)
        with patch.object(c,'ReceiptDestination',Destination),patch.object(c,'convert',side_effect=work),patch.object(c,'emit_receipt',side_effect=report),patch.object(sys,'argv',['tool','--artifact-name','synthetic','--authorize-cpu-conversion','--receipt','unused']),patch.object(sys,'stdout',out),patch.object(sys,'stderr',err):
            code=c.main()
        return code,json.loads(out.getvalue() or err.getvalue()),observed

    def test_cleanup_preserves_publication(self):
        for exc in (KeyboardInterrupt('cleanup'),SystemExit(9),OSError('close')):
            code,state,observed=self.run_case(True,cleanup=exc)
            self.assertEqual(code,1);self.assertEqual(state['promoted'],observed['published'])
            self.assertEqual(state['durability'],'confirmed');self.assertEqual(state['phase'],'receipt_cleanup')

    def test_outer_conversion_cleanup_preserves_shared_state(self):
        code,state,observed=self.run_case(True,work_error=KeyboardInterrupt('outer directory cleanup'))
        self.assertTrue(state['promoted']);self.assertEqual(state['durability'],'confirmed')

    def test_failure_before_publication_not_promoted(self):
        _,state,observed=self.run_case(False,work_error=OSError('before rename'))
        self.assertEqual(state['promoted'],observed['published'])

    def test_reporting_failure_keeps_original(self):
        _,state,_=self.run_case(True,work_error=KeyboardInterrupt('original'),cleanup=SystemExit(8),report_error=OSError('stdout'))
        self.assertEqual(state['error'],'original');self.assertTrue(state['promoted'])
        self.assertEqual(state['cleanup_error_type'],'SystemExit');self.assertEqual(state['reporting_error_type'],'OSError')
