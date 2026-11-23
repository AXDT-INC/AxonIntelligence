"""Shared format/integrity I/O only. No numerical conversion or ML imports."""
import hashlib
import json
import math
import os
import re
import stat
import struct

SOURCE = '/data/axonai/models/merged/AxonAI-v1-27B'
SOURCE_SHA = 'd34c6389c0ca68df59e11f2a18f3616e7b335066fd8c493acf46044551e1daee'
CONVERTED = '/data/axonai/models/converted'
MANIFEST = 'SHA256_MANIFEST.json'
OUTPUT_MANIFEST = 'OUTPUT_SHA256_MANIFEST.json'
COPY_FILES = ('generation_config.json', 'tokenizer_config.json', 'tokenizer.json',
              'chat_template.jinja', 'BASE_LICENSE.Apache-2.0.txt', 'NOTICE.txt', 'PROVENANCE.json')
REQUIRED = set(COPY_FILES) | {'config.json', 'model.safetensors.index.json'}
CHUNK = 128 * 1024  # raw bytes; at most 65,536 values per block
MAX_JSON = 16 * 1024 * 1024
MAX_HEADER = 4 * 1024 * 1024
MAX_TENSORS = 10000
SHARD = re.compile(r'model-[0-9]{5}-of-[0-9]{5}\.safetensors\Z')

class AuditError(Exception):
    pass

def demand(ok, message):
    if not ok:
        raise AuditError(message)

def basename(name):
    demand(isinstance(name, str) and name not in ('', '.', '..') and
           '/' not in name and '\\' not in name and '\x00' not in name, 'unsafe filename')
    return name

def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        demand(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result

def parse_json(data):
    demand(len(data) <= MAX_JSON, 'JSON exceeds bound')
    try:
        return json.loads(data, object_pairs_hook=no_duplicates,
                          parse_constant=lambda x: (_ for _ in ()).throw(AuditError('nonfinite JSON')))
    except (UnicodeError, ValueError) as exc:
        raise AuditError('invalid JSON') from exc

def canonical(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

class Directory:
    """Pin directory descriptors; refuse symlinks in every path component.

    Child files are opened with O_NOFOLLOW. Output writes are O_EXCL. Ancestor
    directories must be trusted against hostile renames by their owners.
    """
    def __init__(self, path=None, *, fd=None):
        if fd is not None:
            self.fd = fd
            return
        path = os.fspath(path)
        demand(os.path.isabs(path), 'directory must be absolute')
        parts = path.split('/')[1:]
        demand(all(p not in ('.', '..') for p in parts), 'unsafe directory components')
        current = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
        try:
            for component in filter(None, parts):
                nxt = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
                os.close(current)
                current = nxt
            self.fd = current
        except BaseException:
            os.close(current)
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        os.close(self.fd)

    def names(self):
        result = set(os.listdir(self.fd))
        for n in result:
            basename(n)
            s = os.stat(n, dir_fd=self.fd, follow_symlinks=False)
            demand(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'not a unique regular file: ' + n)
        return result

    def reader(self, name):
        fd = os.open(basename(name), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.fd)
        s = os.fstat(fd)
        if not (stat.S_ISREG(s.st_mode) and s.st_nlink == 1):
            os.close(fd)
            raise AuditError('not a unique regular file: ' + name)
        return os.fdopen(fd, 'rb')

    def writer(self, name):
        fd = os.open(basename(name), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=self.fd)
        return os.fdopen(fd, 'wb')

    def read(self, name, limit=MAX_JSON):
        with self.reader(name) as f:
            data = f.read(limit + 1)
        demand(len(data) <= limit, 'file exceeds read bound: ' + name)
        return data

    def write(self, name, data):
        with self.writer(name) as f:
            write_all(f,data)
            f.flush()
            os.fsync(f.fileno())

    def fingerprint(self, name):
        with self.reader(name) as f:
            before = os.fstat(f.fileno())
            h = hashlib.sha256()
            size = 0
            while True:
                b = f.read(CHUNK)
                if not b:
                    break
                size += len(b)
                h.update(b)
            after = os.fstat(f.fileno())
        demand(signature(before) == signature(after), 'file changed while hashing: ' + name)
        return {'relative_path': name, 'bytes': size, 'sha256': h.hexdigest()}

    def snapshot(self):
        return {n: signature(os.stat(n, dir_fd=self.fd, follow_symlinks=False)) for n in self.names()}

    def child(self, name):
        return Directory(fd=os.open(basename(name), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                    dir_fd=self.fd))

    def mkdir(self, name):
        os.mkdir(basename(name), 0o700, dir_fd=self.fd)
        return self.child(name)

def signature(s):
    # atime may legitimately change on read; content/identity metadata may not.
    return (s.st_dev, s.st_ino, s.st_mode, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

def checked_manifest(directory, name, expected_sha=None):
    raw = directory.read(name)
    if expected_sha is not None:
        demand(sha_bytes(raw) == expected_sha, 'manifest anchor mismatch')
    m = parse_json(raw)
    demand(isinstance(m, dict) and isinstance(m.get('files'), list) and len(m['files']) <= 10000, 'invalid manifest')
    entries = {}
    for e in m['files']:
        demand(isinstance(e, dict) and set(e) == {'relative_path', 'bytes', 'sha256'}, 'invalid manifest entry')
        n = basename(e['relative_path'])
        demand(n != name and n not in entries, 'duplicate/self manifest entry')
        demand(type(e['bytes']) is int and e['bytes'] >= 0, 'invalid byte count')
        demand(isinstance(e['sha256'], str) and re.fullmatch('[0-9a-f]{64}', e['sha256']), 'invalid SHA256')
        entries[n] = e
    demand(directory.names() == set(entries) | {name}, 'unexpected or missing files')
    for n, e in sorted(entries.items()):
        demand(directory.fingerprint(n) == e, 'hash/size mismatch: ' + n)
    demand(m.get('payload_bytes') == sum(e['bytes'] for e in entries.values()), 'payload total mismatch')
    return m, raw

def read_header(directory, name, dtype):
    with directory.reader(name) as f:
        prefix = f.read(8)
        demand(len(prefix) == 8, 'truncated safetensors length')
        size = struct.unpack('<Q', prefix)[0]
        demand(2 <= size <= MAX_HEADER and size % 8 == 0, 'invalid safetensors header size')
        raw = f.read(size)
        demand(len(raw) == size and raw.startswith(b'{'), 'truncated/invalid safetensors header')
        h = parse_json(raw)
        demand(isinstance(h, dict), 'invalid safetensors header')
        meta = h.get('__metadata__', {})
        demand(isinstance(meta, dict) and all(isinstance(k, str) and isinstance(v, str) for k,v in meta.items()),
               'invalid safetensors metadata')
        tensors = {k: v for k, v in h.items() if k != '__metadata__'}
        demand(0 < len(tensors) <= MAX_TENSORS, 'invalid tensor count')
        end = 0
        total = 0
        for key, v in sorted(tensors.items(), key=lambda kv: tuple(kv[1].get('data_offsets', [0])) if isinstance(kv[1], dict) else (0,)):
            demand(isinstance(key, str) and 0 < len(key) <= 512, 'invalid tensor name')
            demand(isinstance(v, dict) and set(v) == {'dtype', 'shape', 'data_offsets'}, 'invalid tensor metadata')
            demand(v['dtype'] == dtype, 'unsupported dtype')
            shape = v['shape']
            demand(isinstance(shape, list) and len(shape) <= 16 and all(type(x) is int and 0 <= x <= 2**40 for x in shape), 'invalid shape')
            count = math.prod(shape)
            offsets = v['data_offsets']
            demand(isinstance(offsets, list) and len(offsets) == 2 and all(type(x) is int for x in offsets), 'invalid offsets')
            a,b = offsets
            demand(a == end and b >= a and b-a == count*2, 'tensor offsets/shape mismatch')
            end = b
            total += count
        demand(os.fstat(f.fileno()).st_size == 8+size+end, 'truncated or trailing safetensors payload')
    return h, 8+size, total

def model_metadata(directory, dtype):
    index = parse_json(directory.read('model.safetensors.index.json'))
    demand(isinstance(index, dict) and isinstance(index.get('weight_map'), dict), 'invalid index')
    mapping = index['weight_map']
    demand(0 < len(mapping) <= MAX_TENSORS, 'invalid index tensor count')
    shards = sorted(set(mapping.values()))
    demand(all(isinstance(n,str) and SHARD.fullmatch(n) for n in shards), 'unsafe shard name')
    demand(set(n for n in directory.names() if n.endswith('.safetensors')) == set(shards), 'shard inventory mismatch')
    found = {}
    headers = {}
    count = 0
    header_bytes = 0
    demand(shards == [f'model-{i:05d}-of-{len(shards):05d}.safetensors' for i in range(1,len(shards)+1)], 'noncanonical shard sequence')
    for n in shards:
        h, start, elements = read_header(directory, n, dtype)
        header_bytes += start
        demand(header_bytes <= MAX_JSON, 'aggregate headers exceed memory bound')
        headers[n] = (h, start, elements)
        count += elements
        for key in h:
            if key == '__metadata__':
                continue
            demand(key not in found, 'duplicate tensor across shards')
            found[key] = n
            demand(len(found) <= MAX_TENSORS, 'aggregate tensor count exceeds bound')
    demand(found == mapping, 'tensor/index mismatch')
    md = index.get('metadata', {})
    demand(md.get('total_size') == count*2 and md.get('total_parameters') == count, 'index totals mismatch')
    return index, headers, count

def validate_source(directory, anchor):
    m, raw = checked_manifest(directory, MANIFEST, anchor)
    demand(m.get('schema') == 'axonai.standalone-model-sha256.v1' and m.get('self_hash_excluded') is True, 'unsupported source manifest')
    demand(REQUIRED <= directory.names(), 'missing required source metadata')
    index, headers, count = model_metadata(directory, 'BF16')
    demand(m.get('weight_shards') == len(headers), 'manifest shard count mismatch')
    config = parse_json(directory.read('config.json'))
    demand(config.get('dtype') == 'bfloat16' and config.get('architectures') == ['Qwen3_5ForCausalLM']
           and config.get('model_type') == 'qwen3_5_text', 'unexpected source architecture/dtype')
    demand(config.get('mamba_ssm_dtype') == 'float32', 'unexpected recurrent computation dtype')
    if 'torch_dtype' in config:
        demand(config['torch_dtype'] == 'bfloat16', 'conflicting source dtype')
    return m, raw, index, headers, count, config

def output_config(config):
    result = dict(config)
    result['dtype'] = 'float16'
    if 'torch_dtype' in result:
        result['torch_dtype'] = 'float16'
    return result

def copy_file(source, target, name):
    with source.reader(name) as src, target.writer(name) as dst:
        while True:
            b = src.read(CHUNK)
            if not b:
                break
            write_all(dst,b)
        dst.flush()
        os.fsync(dst.fileno())

def make_output_manifest(directory):
    entries = [directory.fingerprint(n) for n in sorted(directory.names())]
    return {'schema': 'axonai.fp16-output-sha256.v1', 'files': entries,
            'payload_bytes': sum(e['bytes'] for e in entries), 'self_hash_excluded': True}

def emit_receipt(result, path=None):
    with ReceiptDestination(path) as destination:
        destination.publish(result)


RECEIPTS = '/data/axonai/receipts'

class ReceiptDestination:
    """Exact external root only; pin its descriptor before any model processing.

    No directories are created and no permission probe writes are performed.
    The root must be private and owned by this effective user. A renamed root
    is detected before and after writing; operations never follow its replacement.
    """
    def __init__(self, path):
        self.directory = None
        self.name = None
        if path is None:
            return  # stdout-only reporting cannot alter protected inventories
        demand(isinstance(path,str) and os.path.isabs(path), 'absolute receipt path required')
        demand(path == os.path.normpath(path), 'receipt path aliases/traversal rejected')
        parent, self.name = os.path.split(path)
        demand(parent == RECEIPTS, 'receipt must be directly inside approved external receipts root')
        basename(self.name)
        with open('/proc/self/mountinfo',encoding='utf-8') as mounts:
            demand(not any(line.split()[4] == RECEIPTS for line in mounts), 'receipt root mount alias rejected')
        d = Directory(RECEIPTS)
        try:
            s = os.fstat(d.fd)
            demand(s.st_uid == os.geteuid() and stat.S_IMODE(s.st_mode) == 0o700,
                   'operator action required: receipts root must be owned by current user with mode 0700')
            demand(os.access(RECEIPTS, os.W_OK | os.X_OK), 'operator action required: receipts root unwritable')
            # Bind-mount aliases of protected roots are rejected by resolved descriptor path.
            resolved = os.readlink('/proc/self/fd/'+str(d.fd))
            demand(resolved == RECEIPTS, 'receipt root alias rejected')
            self.identity = (s.st_dev,s.st_ino)
            self.directory = d
            self.check_identity()
            try:
                os.stat(self.name,dir_fd=d.fd,follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise AuditError('receipt target already exists')
        except BaseException:
            os.close(d.fd)
            self.directory = None
            raise

    def check_identity(self):
        with Directory(RECEIPTS) as current:
            s=os.fstat(current.fd)
            demand((s.st_dev,s.st_ino)==self.identity, 'receipt directory substitution detected')

    def __enter__(self):
        return self

    def __exit__(self,*args):
        if self.directory is not None:
            os.close(self.directory.fd)

    def publish(self,result):
        if self.directory is not None:
            self.check_identity()
            self.directory.write(self.name,canonical(result))
            os.fsync(self.directory.fd)
            self.check_identity()
        print(canonical(result).decode(),end='')

class LifecycleError(AuditError):
    def __init__(self,state,error):
        self.state=dict(state,status='FAIL',error=str(error),error_type=type(error).__name__,interrupted=isinstance(error,(KeyboardInterrupt,SystemExit)))
        super().__init__(str(error))

def write_all(file,data):
    view=memoryview(data)
    while view:
        count=file.write(view)
        demand(type(count) is int and 0 < count <= len(view), 'short/zero write')
        view=view[count:]
