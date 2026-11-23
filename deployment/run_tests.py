#!/usr/bin/env python3
"""Run synthetic tests only and optionally emit an exclusive machine-readable report."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import resource
import sys
import unittest
from audit_io import Directory, canonical

ROOT = Path(__file__).absolute().parent

class Results(unittest.TextTestResult):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.passed = []
    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed.append(test.id())

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-name', help='new basename within deployment only')
    a = p.parse_args()
    # No network event or subprocess is needed by any test in this package.
    def isolation(event, args):
        if event.startswith('socket.') or event in ('subprocess.Popen', 'os.system'):
            raise RuntimeError('network/subprocess forbidden in CPU test suite')
    sys.addaudithook(isolation)
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result = unittest.TextTestRunner(verbosity=2, resultclass=Results).run(suite)
    report = {'status':'PASS' if result.wasSuccessful() else 'FAIL', 'tests_run':result.testsRun,
              'passed':sorted(result.passed), 'failures':[t.id() for t,_ in result.failures],
              'errors':[t.id() for t,_ in result.errors], 'skipped':result.skipped,
              'python':sys.version, 'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'test_dependencies':{n:importlib.metadata.version(n) for n in ('PyYAML','packaging')},
              'fixture_root':str(ROOT/'test-workspace'), 'production_weight_payloads_read':False,
              'network_access':False, 'docker_executed':False, 'gpu_access':False,
              'official_safetensors_library_validation':False,
              'runtime_python_3_12_tested':sys.version_info[:2] == (3,12)}
    if a.report_name:
        with Directory(str(ROOT)) as d:
            d.write(a.report_name, canonical(report))
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(main())
