"""Durable, offline, sequential two-arm benchmark on one reserved L40S pair.

Gold keys and the ledger remain on the host. Workers receive only one prompt at a
 time over stdin. A committed claim is never dispatched twice, even after a crash.
"""
import argparse
import datetime
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import selectors
import sqlite3
import subprocess
import time

from public_suite import canonical, sha, analyze

CODE=Path(__file__).resolve().parent
REPO=CODE.parents[2]
ARMS=('qwen','axonai')


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def emit(**data): print(json.dumps(dict(time=now(),**data),sort_keys=True),flush=True)


class Ledger:
    def __init__(self, root, freeze):
        self.root=Path(root);self.key_path=self.root/'ledger.key'
        if not self.key_path.exists():
            with self.key_path.open('xb') as f: f.write(os.urandom(32))
            self.key_path.chmod(0o600)
        self.key=self.key_path.read_bytes()
        self.db=sqlite3.connect(self.root/'attempts.sqlite3')
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS attempts(item_id TEXT,arm TEXT,block INTEGER,request_sha256 TEXT,
        claim_hmac TEXT,state TEXT,claimed_at TEXT,finished_at TEXT,outcome TEXT,PRIMARY KEY(item_id,arm));''')
        frozen=json.dumps(freeze,sort_keys=True)
        old=self.db.execute("SELECT value FROM metadata WHERE key='freeze'").fetchone()
        if old is not None and old[0]!=frozen: raise RuntimeError('Run identity changed')
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES('freeze',?)",(frozen,));self.db.commit()
        # An unfinished claim may already have reached the model. Never repeat it.
        for ident,arm,block,request,signature in self.db.execute('SELECT item_id,arm,block,request_sha256,claim_hmac FROM attempts'):
            expected=hmac.new(self.key,canonical([ident,arm,block,request]),hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected,signature):raise RuntimeError('Claim mapping signature mismatch')
        self.db.execute("UPDATE attempts SET state='missing_interrupted',finished_at=? WHERE state='claimed'",(now(),));self.db.commit()

    def claim(self,row,arm,job):
        request=hashlib.sha256(canonical(job)).hexdigest()
        signature=hmac.new(self.key,canonical([row['id'],arm,row['block'],request]),hashlib.sha256).hexdigest()
        with self.db:
            cursor=self.db.execute('INSERT OR IGNORE INTO attempts VALUES(?,?,?,?,?,?,?,?,?)',
                (row['id'],arm,row['block'],request,signature,'claimed',now(),None,None))
        return cursor.rowcount==1

    def finish(self,ident,arm,outcome):
        with self.db:
            changed=self.db.execute("UPDATE attempts SET state='completed',finished_at=?,outcome=? WHERE item_id=? AND arm=? AND state='claimed'",(now(),json.dumps(outcome,sort_keys=True),ident,arm)).rowcount
        if changed!=1:raise RuntimeError('Outcome lacks unique active claim')

    def counts(self): return dict(self.db.execute('SELECT state,count(*) FROM attempts GROUP BY state'))


def code_hashes():
    paths=[*CODE.glob('*.py'),*CODE.glob('compose*.yaml'),REPO/'deployment/v100-inference/infer.py',REPO/'deployment/v100-inference/placement.py',REPO/'deployment/v100-inference/score_cases.py']
    return {str(p.relative_to(REPO)):sha(p) for p in sorted(paths)}


def active_gpu_processes(pair):
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    rows=[]
    for line in raw.splitlines():
        if not line.strip():continue
        gpu,pid=[x.strip() for x in line.split(',')]
        if gpu in pair: rows.append((gpu,pid))
    return rows


def inspect_container(name, image, model):
    data=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
    host=data['HostConfig'];cfg=data['Config']
    assert data['Image']==image and cfg['User']=='1000:1000'
    assert host['NetworkMode']=='none' and host['ReadonlyRootfs'] and not host['Privileged']
    assert host['CapDrop']==['ALL'] and any('no-new-privileges' in x for x in host['SecurityOpt'])
    assert host['Memory']==72*(1<<30) and host['MemorySwap']==host['Memory']
    mounts={x['Destination']:x for x in data['Mounts']}
    expected={'/model','/worker/worker.py','/worker/infer.py','/worker/placement.py','/worker/score_cases.py'}
    assert {k for k,v in mounts.items() if v['Type']=='bind'}==expected
    assert all(not mounts[k]['RW'] for k in expected)
    assert mounts['/model']['Source']==str(model)
    return data


class Worker:
    def __init__(self,root,config,arm,block,mode='stream'):
        self.root=root;self.arm=arm;self.config=config
        self.name=f'axonai-l40s-{mode}-{block}-{arm}-{os.getpid()}'
        env=os.environ.copy();env.update(BENCH_MODEL=config['models'][arm],BENCH_IMAGE=config['image'],BENCH_GPU_FIRST=config['gpu_uuids'][0],BENCH_GPU_SECOND=config['gpu_uuids'][1],BENCH_TIMEOUT='48h')
        command=['docker','compose','-p','axonai-l40s-public-benchmark']
        for name in ['compose.yaml','compose.gpu.yaml','compose.l40s.yaml']:command+=['-f',str(CODE/name)]
        command+=['run','--rm','-T','--name',self.name,'worker',arm,mode,'--device','cuda']
        self.log=(root/f'{mode}-{block}-{arm}.jsonl').open('a');self.err=(root/f'{mode}-{block}-{arm}.stderr').open('a')
        self.proc=subprocess.Popen(command,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err)
        self.selector=selectors.DefaultSelector();self.selector.register(self.proc.stdout,selectors.EVENT_READ);self.buffer=b''
        self.verified=None
        try:
            deadline=time.monotonic()+30
            while time.monotonic()<deadline:
                check=subprocess.run(['docker','inspect',self.name],capture_output=True)
                if check.returncode==0:break
                if self.proc.poll() is not None: raise RuntimeError('Container failed before inspection')
                time.sleep(.2)
            receipt=inspect_container(self.name,config['image'],Path(config['models'][arm]))
            (root/f'{mode}-{block}-{arm}.container.json').write_text(json.dumps(receipt,indent=2)+'\n')
            while True:
                item=self.read(900)
                if item.get('event')=='verified':self.verified=item
                if item.get('event')=='loaded':break
            assert self.verified and self.verified['hardware_profile']=='l40s'
            assert self.verified['gpu_uuids']==config['gpu_uuids']
            assert self.verified['manifest']==config['manifests'][arm]
            self.allowed_pids={pid for gpu,pid in active_gpu_processes(config['gpu_uuids'])}
            assert len(self.allowed_pids)==1, 'GPU pair is not exclusive'
        except BaseException:
            self.close();raise

    def read(self,timeout):
        deadline=time.monotonic()+timeout
        while b'\n' not in self.buffer:
            if not self.selector.select(max(0,deadline-time.monotonic())):raise TimeoutError('Worker response deadline')
            data=os.read(self.proc.stdout.fileno(),65536)
            if not data:raise RuntimeError('Worker ended without expected event')
            self.buffer+=data
            if time.monotonic()>deadline:raise TimeoutError('Worker response deadline')
        line,self.buffer=self.buffer.split(b'\n',1)
        item=json.loads(line)
        self.log.write(line.decode()+'\n');self.log.flush();os.fsync(self.log.fileno())
        return item

    def request(self,job):
        self.proc.stdin.write(canonical(job)+b'\n');self.proc.stdin.flush()
        out=self.read(240)
        assert out['event']=='outcome' and out['id']==job['id'] and out['arm']==self.arm
        assert out['manifest']==self.config['manifests'][self.arm]
        return out

    def check_exclusive(self):
        assert {pid for gpu,pid in active_gpu_processes(self.config['gpu_uuids'])}<=self.allowed_pids, 'New GPU co-tenant'

    def close(self):
        if hasattr(self,'proc'):
            if self.proc.stdin and not self.proc.stdin.closed:self.proc.stdin.close()
            try:self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                subprocess.run(['docker','stop','-t','10',self.name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                self.proc.wait(timeout=30)
        for name in ['log','err']:
            if hasattr(self,name):getattr(self,name).close()
        if hasattr(self,'selector'):self.selector.close()


def qualify(root,config):
    receipts={}
    for arm in ARMS:
        assert not active_gpu_processes(config['gpu_uuids']), 'GPU pair busy'
        worker=Worker(root,config,arm,'qualification',mode='qualify')
        try:
            while True:
                item=worker.read(900)
                if item.get('event')=='qualified':
                    receipts[arm]=dict(qualified=item,verified=worker.verified);break
            worker.close()
            assert worker.proc.returncode==0
        finally:worker.close()
        emit(event='qualified',arm=arm,stress_seconds=item['stress_elapsed_s'])
    a,b=[receipts[x] for x in ARMS]
    assert a['qualified']['input_ids_sha256']==b['qualified']['input_ids_sha256']
    assert a['verified']['device_map']==b['verified']['device_map']
    receipt={'completed':now(),'config':config,'code_hashes':code_hashes(),'arms':receipts}
    (root/'qualification.json').write_text(json.dumps(receipt,indent=2)+'\n')


def collect(root,config):
    qualification=json.loads((root/'qualification.json').read_bytes())
    assert qualification['config']==config and qualification['code_hashes']==code_hashes(), 'Qualification identity drift'
    freeze=json.loads((root/'roster-freeze.json').read_bytes())
    assert freeze['roster_sha256']==sha(root/'roster.jsonl')
    assert (root/'screening.json').is_file(), 'Screen before dispatch'
    ledger=Ledger(root,{'config':config,'roster':freeze,'code':code_hashes(),'qualification':sha(root/'qualification.json'),'screening':sha(root/'screening.json')})
    rows=[json.loads(line) for line in (root/'roster.jsonl').read_text().splitlines()]
    for block,arms in enumerate(freeze['block_arm_order']):
        for arm in arms:
            pending=[r for r in rows if r['block']==block and r['eligible'] and not ledger.db.execute('SELECT 1 FROM attempts WHERE item_id=? AND arm=?',(r['id'],arm)).fetchone()]
            if not pending:continue
            assert not active_gpu_processes(config['gpu_uuids']), 'GPU pair busy'
            assert qualification['code_hashes']==code_hashes(), 'Code changed during collection'
            emit(event='loading_block',block=block,arm=arm,items=len(pending))
            worker=Worker(root,config,arm,block)
            try:
                assert worker.verified['device_map']==qualification['arms'][arm]['verified']['device_map']
                for i,row in enumerate(pending):
                    if i%25==0:worker.check_exclusive()
                    job={'id':row['id'],'prompt':row['prompt'],'max_new_tokens':256}
                    if not ledger.claim(row,arm,job):continue
                    outcome=worker.request(job)
                    assert outcome['input_ids_sha256']==row['input_ids_sha256'] and outcome['input_tokens']==row['input_tokens'], 'Prompt token drift'
                    ledger.finish(row['id'],arm,outcome)
                    if (i+1)%25==0:
                        emit(event='progress',block=block,arm=arm,block_done=i+1,block_total=len(pending),counts=ledger.counts())
                worker.close()
                assert worker.proc.returncode==0
            finally:worker.close()
            emit(event='block_complete',block=block,arm=arm,counts=ledger.counts())
    result=analyze(root)
    complete={'completed':now(),'counts':ledger.counts(),'results_sha256':sha(root/'results.json'),'all_slots_attempted':sum(ledger.counts().values())==2*sum(freeze['eligible'].values())}
    (root/'completion.json').write_text(json.dumps(complete,indent=2)+'\n');emit(event='complete',**complete)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['qualify','collect']);parser.add_argument('root',type=Path);args=parser.parse_args()
    os.umask(0o077);root=args.root.resolve();config=json.loads((root/'run-config.json').read_bytes())
    locks=[]
    for uuid in sorted(config['gpu_uuids']):
        f=open('/tmp/axonai-benchmark-'+uuid+'.lock','a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    if args.command=='qualify':qualify(root,config)
    else:collect(root,config)


if __name__=='__main__':main()
