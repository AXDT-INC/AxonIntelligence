import sys
import unittest
from pathlib import Path
import candidate
import convert_fp16 as converter
import native_engine as engine
import test_conversion as fixtures
from audit_io import LifecycleError

class PublicationBoundary(unittest.TestCase):
    def exercise(self, code, needle, mismatch=False):
        fixture = fixtures.ConversionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        state = {}
        artifact = fixture.parent / 'boundary-result'
        line = next(i for i, s in enumerate(Path(code.co_filename).read_text().splitlines(), 1) if needle in s)
        seen = []
        def trace(frame, event, arg):
            if frame.f_code is code and frame.f_lineno == line and event == 'line' and not seen:
                seen.append(artifact.exists())
                with self.assertRaises(engine.WorkerError):
                    engine.complete_publication(dict(state) if mismatch else state)
                self.assertFalse(worker.control(4)['cancellation_complete'])
            return trace
        with self.assertRaises(LifecycleError):
            with candidate.engine('native') as worker:
                try:
                    sys.settrace(trace)
                    converter.convert(str(fixture.source), str(fixture.parent), artifact.name, fixture.anchor, 0, state)
                finally:
                    sys.settrace(None)
        self.assertEqual(seen, [False])
        self.assertFalse(artifact.exists())
    def test_lifecycle_mismatch_before_entry(self):
        self.exercise(converter.convert.__code__, "state.update(phase='atomic_promotion'", True)
    def test_after_python_entry(self):
        self.exercise(engine.perform_promotion.__code__, 'return operation(*args)')
    def test_at_native_entry(self):
        self.exercise(converter.promote.__code__, 'if fn(parent.fd')

    def after_rename(self, needle):
        fixture = fixtures.ConversionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        state = {}
        artifact = fixture.parent / 'post-rename-result'
        line = next(i for i, s in enumerate(Path(converter.__file__).read_text().splitlines(), 1) if needle in s)
        observed = []
        def trace(frame, event, arg):
            if frame.f_code is converter.convert.__code__ and frame.f_lineno == line and event == 'line' and not observed:
                observed.append(artifact.exists())
                with self.assertRaises(engine.WorkerError):
                    engine.complete_publication(state)
            return trace
        with self.assertRaises(engine.WorkerError):
            with candidate.engine('native'):
                try:
                    sys.settrace(trace)
                    converter.convert(str(fixture.source), str(fixture.parent), artifact.name, fixture.anchor, 0, state)
                finally:
                    sys.settrace(None)
        self.assertEqual(observed, [True])
        self.assertTrue(artifact.exists())
        self.assertIs(state['promoted'], True)
        self.assertEqual(state['durability'], 'confirmed')

    def test_completion_after_actual_rename(self):
        self.after_rename("state.update(phase='post_promotion_fsync'")

    def test_completion_during_durability_reporting(self):
        self.after_rename("state.update(phase='published'")
