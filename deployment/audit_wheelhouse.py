#!/usr/bin/env python3
"""Offline wheel metadata closure auditor. Does not install, import or extract wheels.

Requires the auditing host's packaging library, not model runtime dependencies.
A generated lock proves artifact hashes/metadata closure for declared tags only;
it does NOT prove binary compatibility or verify the container base image.
"""
import argparse
from collections import deque
from email.parser import BytesParser
import os
import sys
import zipfile
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from audit_io import Directory, canonical, demand, emit_receipt

ENVIRONMENT = {'implementation_name':'cpython', 'implementation_version':'3.12.0',
               'os_name':'posix', 'platform_machine':'x86_64', 'platform_release':'',
               'platform_system':'Linux', 'platform_version':'', 'python_full_version':'3.12.0',
               'platform_python_implementation':'CPython', 'python_version':'3.12', 'sys_platform':'linux'}

def audit(wheelhouse, requirements, accepted_tags):
    demand(accepted_tags, 'explicit target wheel tags required')
    with Directory(wheelhouse) as directory:
        before = directory.snapshot()
        inventory = {}
        for filename in sorted(directory.names()):
            demand(filename.endswith('.whl'), 'unexpected non-wheel file')
            wheel_name, version, _, tags = parse_wheel_filename(filename)
            demand({str(t) for t in tags} & set(accepted_tags), 'wheel target tag not approved: '+filename)
            receipt = directory.fingerprint(filename)
            with directory.reader(filename) as file, zipfile.ZipFile(file) as wheel:
                for n in wheel.namelist():
                    demand(not n.startswith('/') and '..' not in n.split('/') and '\\' not in n, 'unsafe wheel member')
                metadata = [x for x in wheel.infolist() if x.filename.endswith('.dist-info/METADATA')]
                demand(len(metadata) == 1 and metadata[0].file_size <= 4*1024**2, 'invalid/big wheel METADATA')
                with wheel.open(metadata[0]) as member:
                    raw = member.read(4*1024**2+1)
                demand(len(raw) <= 4*1024**2, 'METADATA decompression bound exceeded')
                data = BytesParser().parsebytes(raw)
            name = canonicalize_name(data.get('Name', ''))
            demand(name == canonicalize_name(wheel_name) and data.get('Version') == str(version), 'wheel identity mismatch')
            demand(name not in inventory, 'ambiguous duplicate distribution: '+name)
            requires_python = data.get('Requires-Python')
            if requires_python:
                from packaging.specifiers import SpecifierSet
                demand(SpecifierSet(requires_python).contains('3.12.0'), 'Python requirement mismatch: '+name)
            inventory[name] = {'name':name, 'version':str(version), 'wheel':filename,
                               'sha256':receipt['sha256'], 'bytes':receipt['bytes'],
                               'requires_python':requires_python,
                               'requires_dist':data.get_all('Requires-Dist', []),
                               'provides_extra':data.get_all('Provides-Extra', [])}
        queue = deque(Requirement(line) for line in requirements.splitlines()
                      if line.strip() and not line.lstrip().startswith('#'))
        seen = set()
        selected = {}
        while queue:
            req = queue.popleft()
            demand(req.url is None, 'direct URL dependencies are not supported offline')
            name = canonicalize_name(req.name)
            demand(name in inventory, 'missing transitive wheel: '+name)
            wheel = inventory[name]
            demand(req.specifier.contains(wheel['version'], prereleases=True), 'dependency version conflict: '+str(req))
            selected[name] = wheel
            for extra in {''} | set(req.extras):
                demand(not extra or extra in wheel['provides_extra'], 'unknown requested extra: '+extra)
                if (name,extra) in seen:
                    continue
                seen.add((name,extra))
                env = dict(ENVIRONMENT, extra=extra)
                for raw in wheel['requires_dist']:
                    dependency = Requirement(raw)
                    if dependency.marker is None or dependency.marker.evaluate(env):
                        queue.append(dependency)
        demand(set(selected) == set(inventory), 'unreferenced wheels in wheelhouse; use a dedicated exact closure')
        for wheel in inventory.values():
            demand(directory.fingerprint(wheel['wheel'])['sha256'] == wheel['sha256'], 'wheel changed during audit')
        demand(directory.snapshot() == before, 'wheelhouse changed during audit')
    lock = '# OFFLINE HASHED WHEEL CLOSURE; runtime/ABI validation still required.\n'
    lock += ''.join(f"{n}=={w['version']} --hash=sha256:{w['sha256']}\n" for n,w in sorted(selected.items()))
    report = {'status':'PASS_METADATA_CLOSURE_ONLY', 'target_environment':ENVIRONMENT,
              'operator_declared_wheel_tags':sorted(accepted_tags),
              'dependencies':[selected[n] for n in sorted(selected)],
              'base_image_digest_verified':False, 'native_library_abi_verified':False,
              'gpu_execution_verified':False}
    return report, lock

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wheelhouse', required=True)
    p.add_argument('--accepted-tag', action='append', required=True)
    p.add_argument('--output-directory', required=True)
    a = p.parse_args()
    try:
        with Directory(os.path.dirname(os.path.abspath(__file__))) as package:
            requirements = package.read('requirements.candidate.txt').decode()
        report, lock = audit(a.wheelhouse, requirements, a.accepted_tag)
        with Directory(a.output_directory) as output:
            for n in ('requirements.lock', 'dependency-audit.resolved.json'):
                try:
                    os.stat(n, dir_fd=output.fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                raise ValueError('audit output already exists: '+n)
            output.write('dependency-audit.resolved.json', canonical(report))
            output.write('requirements.lock', lock.encode())
        emit_receipt(report)
        return 0
    except Exception as exc:
        emit_receipt({'status':'FAIL', 'phase':'offline-dependency-audit', 'error':str(exc)})
        return 1

if __name__ == '__main__':
    sys.exit(main())
