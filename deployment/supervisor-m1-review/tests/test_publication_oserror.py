import os
import sys
import unittest
from unittest.mock import patch

import candidate
import convert_fp16
from audit_io import LifecycleError
import test_conversion


class PublicationOSError(unittest.TestCase):
    def after_rename(self, selection):
        fixture = test_conversion.ConversionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        lifecycle = {}
        def interrupt(frame, event, arg):
            if frame.f_code is convert_fp16.promote.__code__ and event == 'return':
                raise OSError('post-rename notification')
            return interrupt
        with self.assertRaises(LifecycleError):
            with candidate.engine(selection) as worker:
                try:
                    sys.settrace(interrupt)
                    convert_fp16.convert(str(fixture.source), str(fixture.parent),
                                         'observed', fixture.anchor, 0, lifecycle)
                finally:
                    sys.settrace(None)
        self.assertTrue((fixture.parent / 'observed' / 'CONVERSION.json').is_file())
        self.assertEqual(lifecycle['primary_failure']['error'], 'post-rename notification')
        if selection == 'native':
            self.assertIs(lifecycle['promoted'], True)
            self.assertEqual(lifecycle['durability'], 'uncertain')
            self.assertEqual(worker.native_publication_status()['confirmed'], 1)
            self.assertEqual(worker.info['commit_state'], 'published')
            self.assertEqual(worker.info['commit_durability'], 1)
        else:
            self.assertIsNone(lifecycle['promoted'])
            self.assertEqual(lifecycle['durability'], 'unknown')

    def test_reference_post_return_oserror_keeps_unknown(self):
        self.after_rename('reference')

    def test_native_post_return_oserror_keeps_confirmation(self):
        self.after_rename('native')

    def test_reference_collision_preserves_definite_syscall_failure(self):
        fixture = test_conversion.ConversionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        promote = convert_fp16.promote

        def collision(parent, stage, final):
            os.mkdir(final, dir_fd=parent.fd)
            return promote(parent, stage, final)

        lifecycle = {}
        with patch.object(convert_fp16, 'promote', collision), self.assertRaises(LifecycleError):
            convert_fp16.convert(str(fixture.source), str(fixture.parent),
                                 'occupied', fixture.anchor, 0, lifecycle)
        self.assertIs(lifecycle['promoted'], False)
        self.assertEqual(lifecycle['durability'], 'not_published')
        self.assertEqual(lifecycle['primary_failure']['error_type'], 'FileExistsError')
        self.assertEqual(list((fixture.parent / 'occupied').iterdir()), [])
        self.assertTrue((fixture.parent / lifecycle['staging_name'] / 'CONVERSION.json').is_file())
