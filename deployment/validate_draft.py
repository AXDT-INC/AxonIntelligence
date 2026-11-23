#!/usr/bin/env python3
"""Offline YAML/security lint. NOT Docker Compose schema or engine validation."""
import json
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).absolute().parent

def check(path=ROOT/'compose.yaml'):
    data = yaml.safe_load(path.read_text())
    assert data['name'] == 'axonai-v1'
    assert set(data['services']) == {'convert','verify','inference','api','rag'}
    assert not data.get('networks') and not data.get('volumes')
    for name, service in data['services'].items():
        assert service['profiles'], (name, 'must be inactive by default')
        assert service['network_mode'] == 'none'
        assert service['user'] == '1000:1000'
        assert service['read_only'] is True and service['privileged'] is False
        assert service['cap_drop'] == ['ALL']
        assert service['security_opt'] == ['no-new-privileges:true']
        assert service['restart'] == 'no' and service['pull_policy'] == 'never'
        assert not any(k in service for k in ('ports','networks','build','devices','container_name','depends_on','ipc','pid'))
        for volume in service.get('volumes', []):
            assert volume['type'] == 'bind' and volume['source'].startswith('/data/axonai/')
            assert volume['bind']['create_host_path'] is False
            if volume['target'] in ('/package','/model','/authorization') or '/merged/' in volume['target']:
                assert volume['read_only'] is True
            if volume['read_only'] is False:
                assert (name, volume['target']) in {('convert','/data/axonai/models/converted'), ('verify','/data/axonai/receipts'), ('convert','/data/axonai/receipts')}
        if name != 'inference':
            assert service['runtime'] == 'runc'
            assert not service.get('deploy')
            assert service['environment']['NVIDIA_VISIBLE_DEVICES'] == 'void'
            assert service['environment']['CUDA_VISIBLE_DEVICES'] == ''
    gpu = data['services']['inference']
    devices = gpu['deploy']['resources']['reservations']['devices']
    assert len(devices) == 1 and devices[0]['driver'] == 'nvidia' and 'count' not in devices[0]
    assert devices[0]['device_ids'] == ['${AXONAI_GPU_UUID_0:-NOT_RESERVED}', '${AXONAI_GPU_UUID_1:-NOT_RESERVED}']
    assert '${AXONAI_GPU_AUTHORIZATION:---authorization-not-granted}' in gpu['command']
    assert '${AXONAI_CPU_AUTHORIZATION:---authorization-not-granted}' in data['services']['convert']['command']
    for name in ('api','rag'):
        assert 'not implemented or authorized' in data['services'][name]['command'][-1]
    return {'status':'PASS_OFFLINE_DRAFT_LINT', 'docker_executed':False,
            'compose_schema_validated_by_docker':False, 'container_images_verified':False,
            'default_active_services':[], 'services':sorted(data['services'])}

if __name__ == '__main__':
    try:
        print(json.dumps(check(), sort_keys=True, indent=2))
    except Exception as exc:
        print(json.dumps({'status':'FAIL','error':str(exc)}, sort_keys=True))
        sys.exit(1)
