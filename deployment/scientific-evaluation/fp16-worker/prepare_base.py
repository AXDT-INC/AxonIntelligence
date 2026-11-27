"""Build the bare FP16 text checkpoint from original Qwen BF16 bytes, offline."""
from pathlib import Path
import hashlib
import fcntl
import json
import math
import os
import struct
import sys
import tempfile

import numpy as np
from safetensors import safe_open
from files import digest, require
from verify import read_header, rounding_table

SOURCE = Path('/source')
LAYOUT = Path('/layout')
OUT = Path('/output')
INDEX = 'model.safetensors.index.json'
LAYOUT_SHA = 'c3a6cb1deb5123cc35ccc9c07de36545f82b91a317f97971b6bd56b3403f2e06'
PROVENANCE_SHA = 'f159c32a878ed623dc7cb98cb48f50d25f1fc5d16368811b8ebe00edcf533e9f'
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
CHUNK = 4 << 20


def source_catalog(root, index):
    catalog = {}
    for shard in sorted(set(index.values())):
        path = root / shard
        with path.open('rb') as stream:
            header, start, _ = read_header(stream)
        header.pop('__metadata__', None)
        require(set(header) == {k for k, v in index.items() if v == shard}, shard)
        for key, info in header.items():
            left, right = info['data_offsets']
            require(0 <= left <= right <= path.stat().st_size - start, key)
            catalog[key] = (path, start + left, right - left, info['dtype'], info['shape'])
    return catalog


def make_shard(path, keys, catalog):
    offset = 0
    header = {'__metadata__': {'format': 'pt'}}
    for key in keys:
        _, _, length, dtype, shape = catalog[key]
        require(dtype == 'BF16' and length == 2 * math.prod(shape), key)
        header[key] = {'dtype': 'F16', 'shape': shape, 'data_offsets': [offset, offset + length]}
        offset += length
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    with path.open('wb') as out:
        out.write(struct.pack('<Q', len(raw)) + raw)
        for key in keys:
            src, offset, remaining, _, _ = catalog[key]
            with src.open('rb') as stream:
                stream.seek(offset)
                while remaining:
                    count = min(CHUNK, remaining)
                    data = stream.read(count)
                    require(len(data) == count, key)
                    fp32 = (np.frombuffer(data, '<u2').astype('<u4') << 16).view('<f4')
                    with np.errstate(over='raise', invalid='raise'):
                        half = fp32.astype('<f2')
                    require(np.isfinite(half).all(), key)
                    out.write(half.tobytes())
                    remaining -= count
        out.flush()
        os.fsync(out.fileno())


def verify_shard(path, keys, catalog, table):
    """Independently decode the saved file and compare every half-word to the struct oracle."""
    bits, valid = table
    with safe_open(path, framework='np') as reader, path.open('rb') as out:
        require(set(reader.keys()) == set(keys) and reader.metadata() == {'format': 'pt'}, path)
        h, start, _ = read_header(out)
        for key in keys:
            src, source_offset, remaining, _, shape = catalog[key]
            dest = h[key]
            lo, hi = dest['data_offsets']
            require(dest['shape'] == shape == reader.get_slice(key).get_shape(), key)
            require(dest['dtype'] == 'F16' and hi - lo == remaining, key)
            out.seek(start + lo)
            with src.open('rb') as original:
                original.seek(source_offset)
                while remaining:
                    count = min(CHUNK, remaining)
                    raw = original.read(count)
                    saved = out.read(count)
                    require(len(raw) == len(saved) == count, key)
                    words = np.frombuffer(raw, '<u2')
                    require(valid[words].all(), key)
                    require(np.array_equal(bits[words], np.frombuffer(saved, '<u2')), key)
                    remaining -= count


def fixture():
    raw = struct.pack('<5H', 0, 0x8000, 0x3f80, 0xc000, 0x3380)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root/'bf16').write_bytes(raw)
        cat = {'test': (root/'bf16', 0, len(raw), 'BF16', [5])}
        metadata = root/'metadata.json'
        metadata.with_suffix('.json.part').write_bytes(b'incomplete')
        save_metadata(metadata, b'{}')
        save_metadata(metadata, b'{}')
        require(not metadata.with_suffix('.json.part').exists(), 'Metadata commit incomplete')
        out = root/'tiny.safetensors'
        table = rounding_table()
        make_shard(out, ['test'], cat)
        verify_shard(out, ['test'], cat, table)
        modified = bytearray(out.read_bytes())
        modified[-1] ^= 1
        out.write_bytes(modified)
        try:
            verify_shard(out, ['test'], cat, table)
        except ValueError:
            print('Synthetic rounding, safetensors readback and corruption detection passed.')
            return
        raise ValueError('Corruption was accepted')


def save_metadata(path, data):
    if path.exists():
        require(path.read_bytes() == data, f'Existing file differs: {path.name}')
    else:
        pending = path.with_name(path.name + '.part')
        with pending.open('wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        pending.rename(path)
    path.chmod(0o400)


def build(provenance_path=Path('/job/PROVENANCE.json')):
    stage = OUT/'.text-fp16.partial'
    target = OUT/'Qwen3.8-27B-text-FP16'
    require(not target.exists(), 'Target already published')
    stage.mkdir(exist_ok=True)
    stage.chmod(0o700)
    prov = Path(provenance_path)
    require(digest(prov) == PROVENANCE_SHA, 'Source inventory identity')
    recorded = json.loads(prov.read_bytes())
    require(recorded['base_revision'] == REVISION, 'Source revision')
    originals = {Path(r['path']).name: r for r in recorded['base_files']
                 if Path(r['path']).parent == Path('/data/axonai/models/base/Qwen3.8-27B')}
    require(len(originals) == 27, 'Original file inventory')
    for name, row in originals.items():
        path = SOURCE/name
        require(path.is_file() and path.stat().st_size == row['bytes'] and digest(path) == row['sha256'], name)
    manifest = LAYOUT/'SHA256_MANIFEST.json'
    require(digest(manifest) == LAYOUT_SHA, 'Release layout manifest digest')
    rows = {r['relative_path']: r for r in json.loads(manifest.read_bytes())['files']}
    shared = [INDEX,'config.json','generation_config.json','tokenizer_config.json','tokenizer.json','chat_template.jinja']
    for name in shared:
        require(digest(LAYOUT/name) == rows[name]['sha256'], name)
    original_cfg = json.loads((SOURCE/'config.json').read_bytes())['text_config']
    text_cfg = json.loads((LAYOUT/'config.json').read_bytes())
    require(original_cfg.pop('dtype') == 'bfloat16' and text_cfg.pop('dtype') == 'float16', 'Precision')
    require(text_cfg.pop('architectures') == ['Qwen3_5ForCausalLM'], 'Text class')
    text_cfg.pop('transformers_version')
    require(original_cfg == text_cfg, 'Text hyperparameters')
    generation_cfg = json.loads((LAYOUT/'generation_config.json').read_bytes())
    generation_cfg.pop('transformers_version')
    require(generation_cfg == json.loads((SOURCE/'generation_config.json').read_bytes()), 'Generation settings')
    original = json.loads((SOURCE/INDEX).read_bytes())['weight_map']
    idx = json.loads((LAYOUT/INDEX).read_bytes())
    text = idx['weight_map']
    dropped = sorted(set(original) - set(text))
    require(len(text) == 851 and len(dropped) == 348 and set(text) <= set(original), 'Projection keys')
    require(len(set(original.values())) == 18 and len(set(text.values())) == 14, 'Shard inventory')
    catalog = source_catalog(SOURCE, original)
    # Digest of the canonical shape map independently extracted from the release headers.
    shapes = {k: catalog[k][4] for k in text}
    encoded = (json.dumps(shapes, sort_keys=True, separators=(',', ':')) + '\n').encode()
    require(hashlib.sha256(encoded).hexdigest() == 'b08148c609ae062b1fbebd6470ea5d851e50a7c9ebbeaff342f083634e37daa5',
            'Cross-arm tensor shapes')
    elements = sum(math.prod(catalog[k][4]) for k in text)
    require(elements == idx['metadata']['total_parameters'] == 26895998464, 'Text parameter count')
    require(2 * elements == idx['metadata']['total_size'], 'Text storage')
    table = rounding_table()
    results = []
    for shard in sorted(set(text.values())):
        keys = sorted(key for key, dest in text.items() if dest == shard)
        path = stage/shard
        if not path.exists():
            partial = stage/(shard+'.part')
            make_shard(partial, keys, catalog)
            verify_shard(partial, keys, catalog, table)
            partial.rename(path)
        else:
            verify_shard(path, keys, catalog, table)
        results.append(dict(relative_path=shard, bytes=path.stat().st_size, sha256=digest(path)))
        path.chmod(0o400)
        print(f'Converted and independently verified {shard} ({len(keys)} tensors)', flush=True)
    for name in shared + ['LICENSE']:
        path = stage/name
        data = ((SOURCE if name == 'LICENSE' else LAYOUT)/name).read_bytes()
        save_metadata(path, data)
        results.append(dict(relative_path=name, bytes=len(data), sha256=digest(path)))
    receipt = dict(source=recorded['base_repository_recorded'], revision=REVISION, source_provenance_sha256=PROVENANCE_SHA,
                   release_layout_manifest_sha256=LAYOUT_SHA, shared_runtime_assets=shared,
                   tensors=len(text), elements=elements, excluded_source_keys=dropped, adapter_modules=0,
                   method='Bare source BF16 -> IEEE FP32 -> FP16, independently checked every saved word against struct binary16',
                   self_hash_excluded=True, files=results)
    manifest = stage/'SHA256_MANIFEST.json'
    data = (json.dumps(receipt, indent=2, sort_keys=True)+'\n').encode()
    save_metadata(manifest, data)
    require({p.name for p in stage.iterdir()} == {r['relative_path'] for r in results} | {manifest.name}, 'Export inventory')
    require(digest(manifest) == '644bca802d87721db6656d3ef4f7bf8e8c79bfdbe2761c4abce5086186b06c6e',
            'Reproduced FP16 manifest differs from worker pin')
    stage.chmod(0o500)
    stage.rename(target)
    print('Verified FP16 manifest SHA-256:', digest(target/manifest.name))


if __name__ == '__main__':
    os.umask(0o077)
    if sys.argv[1:] == ['fixture']:
        fixture()
    elif sys.argv[1:] == ['export']:
        require(not OUT.samefile(SOURCE) and not OUT.samefile(LAYOUT), 'Preparation output overlaps a model')
        with os.fdopen(os.open(OUT / '.prepare.lock', os.O_CREAT | os.O_RDWR, 0o600), 'r+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            build()
    else:
        raise SystemExit('Usage: export.py fixture|export')
