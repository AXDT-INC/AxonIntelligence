#!/usr/bin/env python3
"""New recovery verifier (2026-11-23), NOT the original assembly program.
Read-only, standard library, streaming digest/comparison. Never writes corpus data.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import re
import sys

PACKAGE = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_LINE_BYTES = 1024 * 1024


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def norm(question):
    # Explicit verifier rule, consistent with production source line 620.
    # Original final-assembly normalization code has NOT been recovered.
    return re.sub(r'\s+', ' ', question).strip().lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', type=Path, default=Path('/data/axonai'))
    args = ap.parse_args()
    root = args.data_root.resolve()
    pins = read_json(PACKAGE / 'evidence/EXTERNAL_INPUTS.json')
    files = {}
    checked = {}
    for name, info in pins.items():
        p = (root / info['original_relative_path']).resolve()
        need(p.is_relative_to(root), f'input escapes data root: {name}')
        need(p.is_file(), f'missing external input: {name}: {p}')
        need(p.stat().st_size == info['bytes'] <= MAX_FILE_BYTES, f'size bound/pin mismatch: {name}')
        h = hashlib.sha256()
        with p.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
        need(h.hexdigest() == info['sha256'], f'digest mismatch: {name}')
        files[name] = p
        checked[name] = {'sha256': h.hexdigest(), 'bytes': info['bytes']}

    def rows(name):
        count = 0
        with files[name].open('rb') as stream:
            while True:
                line = stream.readline(MAX_LINE_BYTES + 1)
                if not line:
                    break
                need(len(line) <= MAX_LINE_BYTES, f'line bound: {name}')
                need(bool(line.strip()), f'blank row: {name}')
                count += 1
                yield json.loads(line)
        need(count == pins[name]['records'], f'count mismatch: {name}')

    plan = list(rows('supersession_rows'))
    excluded = {x['supersede_record_id'] for x in plan}
    observed = read_json(PACKAGE / 'evidence/OCT09_OBSERVATION.json')
    historical_exclusions = next(x['observed_json'] for x in observed['blocks'] if x['original_relative_path'].endswith('superseded-production-record-ids.json'))
    need(excluded == set(historical_exclusions) and len(excluded) == 12, 'supersession plan/final IDs differ')
    routing = {x['slot_id']: (i, x['domain'], x['task_type']) for i, x in enumerate(rows('routing7200'))}
    need(len(routing) == 7200, 'routing slot uniqueness')
    exhausted = {x['slot_id'] for x in rows('exhausted')}
    need(len(exhausted) == 28, 'exhausted slot uniqueness')
    sources = {x['slot_id']: x for x in rows('source12')}
    sources.update({x['slot_id']: x for x in rows('source1')})
    need(len(sources) == 12, 'replacement source slot count')
    all_meta, final_ids, questions = {}, set(), set()
    baseline_questions = Counter()
    domains, tasks, issuers, groups = Counter(), Counter(), Counter(), {}
    production_slots, recovery_slots, replacement_slots = [], [], []
    digest = hashlib.sha256()
    n = byte_count = 0
    with files['staging'].open('rb') as target:
        for name in ['batch001', 'pmc', 'production', 'recovered28', 'previous11', 'single']:
            count = 0
            start = n + 1
            for row in rows(name):
                rid, prov = row['id'], row['provenance']
                question = next(m['content'] for m in row['messages'] if m['role'] == 'user')
                q = norm(question)
                all_meta[rid] = (row['domain'], row['task_type'], q, prov.get('frozen_slot_id'))
                if name in ['batch001', 'pmc', 'production']:
                    baseline_questions[q] += 1
                if name == 'production':
                    production_slots.append(prov['frozen_slot_id'])
                    if rid in excluded:
                        continue
                if name == 'recovered28':
                    recovery_slots.append(prov['frozen_slot_id'])
                if name in ['previous11', 'single']:
                    slot = prov['frozen_slot_id']
                    replacement_slots.append(slot)
                    old = prov['supersedes_production_record_id']
                    need(old in excluded, 'replacement supersession not in plan')
                    need((row['domain'], row['task_type'], slot) == (all_meta[old][0], all_meta[old][1], all_meta[old][3]), 'replacement changes domain/task/slot')
                    source = sources[slot]
                    need(prov['source_unit_id'] == source['source_unit_id'] and prov['source_sha256'] == source['source_sha256'], 'replacement source identity mismatch')
                    expected_manifest = pins['source1' if name == 'single' else 'source12']['sha256']
                    need(prov['recovery_manifest_sha256'] == expected_manifest, 'replacement source manifest binding')
                need(rid not in final_ids and q not in questions, 'duplicate final ID or normalized question')
                final_ids.add(rid)
                questions.add(q)
                domains[row['domain']] += 1
                tasks[(row['domain'], row['task_type'])] += 1
                issuers[row['source']] += 1
                encoded = (json.dumps(row, ensure_ascii=False, separators=(',', ':'), sort_keys=False) + '\n').encode('utf-8')
                need(target.readline(MAX_LINE_BYTES + 1) == encoded, f'byte/order mismatch at output row {n+1} ({name})')
                digest.update(encoded)
                byte_count += len(encoded)
                n += 1
                count += 1
            groups[name] = {'records': count, 'first_row_1based': start, 'last_row_1based': n}
        need(target.read(1) == b'', 'extra staging bytes')
    need(n == 20000 and digest.hexdigest() == pins['staging']['sha256'], 'final reconstruction digest/count')
    need(len(production_slots) == len(set(production_slots)) == 7172, 'production slot uniqueness')
    need([routing[s][0] for s in production_slots] == sorted(routing[s][0] for s in production_slots), 'production manifest order')
    need(set(production_slots).isdisjoint(exhausted) and set(production_slots) | exhausted == set(routing), 'production/exhaustion coverage')
    need(len(recovery_slots) == 28 and set(recovery_slots) == exhausted, '28 recovery slot coverage')
    need(len(replacement_slots) == 12 and set(replacement_slots) == {x['replacement_slot_id'] for x in plan}, 'replacement slot coverage')
    need(sum(c-1 for c in baseline_questions.values()) == 12, 'baseline duplicate surplus')
    for p in plan:
        old, retained = all_meta[p['supersede_record_id']], all_meta[p['retain_record_id']]
        need(old[2] == retained[2] and p['retain_record_id'] in final_ids, 'planned duplicate retention')
    historical_audit = next(x['observed_json'] for x in observed['blocks'] if x['original_relative_path'].endswith('staging-audit-report.json'))
    need(dict(domains) == historical_audit['domain_counts'], 'domain audit mismatch')
    need(dict(issuers) == historical_audit['source_issuer_counts'], 'issuer audit mismatch')
    # Manifests adjacent to intermediate payloads supply separate file-level evidence.
    native_bindings = {}
    for name, path in files.items():
        sums = path.parent / 'SHA256SUMS'
        if sums.is_file():
            for line in sums.read_text().splitlines():
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and Path(parts[1].lstrip('*')).name == path.name:
                    need(parts[0] == pins[name]['sha256'], f'historical member digest mismatch: {name}')
                    native_bindings[name] = str(sums.relative_to(root))
    print(json.dumps({
        'status': 'PASS_BYTE_EXACT_COMPOSITION_FROM_RETAINED_INTERMEDIATES',
        'original_assembly_implementation_recovered': False,
        'original_assembly_invocation_recovered': False,
        'full_raw_to_20k_historical_replay_proven': False,
        'python': platform.python_version(),
        'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'external_inputs_verified': checked,
        'historical_member_checksum_bindings_verified': native_bindings,
        'composition_order': groups,
        'records': n, 'unique_ids': len(final_ids), 'unique_questions_whitespace_lowercase': len(questions),
        'baseline_exact_question_duplicate_groups': sum(c>1 for c in baseline_questions.values()),
        'baseline_duplicate_extra_records': sum(c-1 for c in baseline_questions.values()),
        'superseded_production_records': len(excluded),
        'replacement_domain_task_slot_and_source_bindings_verified': True,
        'production_order_matches_routing_manifest': True,
        'recovered28_exactly_covers_exhausted_slots': True,
        'domain_counts': dict(domains), 'source_issuer_counts': dict(issuers),
        'domain_task_counts': {d+'|'+t: c for (d,t),c in sorted(tasks.items())},
        'reconstructed_bytes': byte_count, 'reconstructed_sha256': digest.hexdigest(),
        'original_staging_bytes_matched_line_by_line': True,
        'serialization': 'json.dumps(row, ensure_ascii=False, separators=(comma, colon), sort_keys=False) + LF; UTF-8',
        'corpus_output_written': False,
        'limitations': ['Original assembly script/invocation and their execution identity are missing.', 'Five original archive pins and the supersession_plan pin remain unverified against original payloads.', 'This verifier does not recreate LLM generation or the historical near-duplicate audit.']
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, StopIteration) as error:
        print('FAIL: '+str(error), file=sys.stderr)
        sys.exit(1)
