"""Read-only aggregate progress; never emit examples, answers, IDs or correctness."""
import argparse
from collections import Counter, defaultdict
import datetime as dt
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'fp16-worker'))
from public_suite import parse_answer

ARMS = ('qwen', 'axonai')


def timestamp(value):
    return dt.datetime.fromisoformat(value)


def inspect_phase(root, now):
    freeze = json.loads((root / 'roster-freeze.json').read_bytes())
    rows = [json.loads(line) for line in (root / 'roster.jsonl').read_text().splitlines()]
    by_id = {r['id']: r for r in rows}
    records = []
    if (root / 'attempts.sqlite3').exists():
        with sqlite3.connect('file:' + str(root / 'attempts.sqlite3') + '?mode=ro', uri=True) as db:
            records = list(db.execute('SELECT item_id,arm,state,claimed_at,finished_at,outcome FROM attempts'))
    stats = {name: {'planned_questions': total, 'eligible_questions': freeze['eligible'].get(name, 0),
                   'scheduled_responses': 2 * freeze['eligible'].get(name, 0),
                   'completed': 0, 'in_flight': 0, 'missing_interrupted': 0,
                   'inference_failures': 0, 'format_failures': 0, 'cap_hits': 0,
                   'arms_completed': dict.fromkeys(ARMS, 0), 'outcome_statuses': Counter()}
             for name, total in freeze['planned'].items()}
    timings = defaultdict(list)
    recent = []
    attempts = {}
    for ident, arm, state, claimed, finished, payload in records:
        row = by_id[ident]; name = row['suite']; summary = stats[name]
        attempts[(ident, arm)] = (state, claimed)
        if state == 'claimed': summary['in_flight'] += 1
        if state == 'missing_interrupted': summary['missing_interrupted'] += 1
        if state != 'completed': continue
        summary['completed'] += 1; summary['arms_completed'][arm] += 1
        out = json.loads(payload); status = out.get('status', 'unknown')
        summary['outcome_statuses'][status] += 1
        if status not in ('ok', 'empty_output'):
            summary['inference_failures'] += 1
        else:
            summary['format_failures'] += parse_answer(out.get('text', ''), name) is None
        summary['cap_hits'] += out.get('stop') == 'cap'
        if out.get('elapsed_s', 0) > 0:
            timings[(name, arm)].append((out['elapsed_s'], out.get('output_tokens', 0)))
        if finished and (now - timestamp(finished)).total_seconds() <= 600: recent.append(finished)
    return freeze, rows, attempts, stats, timings, len(recent)


def report(parent, now=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    phases = [parent]
    if (parent / 'gpqa-extension/roster-freeze.json').exists(): phases.append(parent / 'gpqa-extension')
    loaded = [inspect_phase(root, now) for root in phases]
    timings = {k: v for _, _, _, _, samples, _ in loaded for k, v in samples.items()}
    qualification = json.loads((parent / 'qualification.json').read_bytes())
    stress = {arm: qualification['arms'][arm]['qualified']['stress_elapsed_s'] for arm in ARMS}
    ratio = stress['axonai'] / stress['qwen']
    all_samples = [sample for samples in timings.values() for sample in samples]
    fallback = sum(x[0] for x in all_samples) / len(all_samples) if all_samples else max(stress.values())

    def estimate(name, arm):
        samples = timings.get((name, arm))
        if samples: return sum(x[0] for x in samples) / len(samples)
        other = 'qwen' if arm == 'axonai' else 'axonai'
        if timings.get((name, other)):
            return estimate(name, other) * (ratio if arm == 'axonai' else 1 / ratio)
        # GPQA has no observations yet: conservatively project a full output cap
        # from observed GSM8K generation throughput, retaining the stress floor.
        gsm = timings.get(('gsm8k', arm)) or timings.get(('gsm8k', other))
        if name == 'gpqa' and gsm:
            scale = 1 if timings.get(('gsm8k', arm)) else ratio if arm == 'axonai' else 1 / ratio
            return max(stress[arm], 256 * sum(x[0] for x in gsm) / max(1, sum(x[1] for x in gsm)) * scale)
        return max(fallback, stress[arm])

    remaining_seconds = 0.; suite_finish = {}; stats = {}; phase_finish = {}
    for root, (freeze, rows, attempts, phase_stats, _, _) in zip(phases, loaded):
        stats.update(phase_stats)
        for block, arms in enumerate(freeze['block_arm_order']):
            for arm in arms:
                pending = [r for r in rows if r['block'] == block and r['eligible']
                           and attempts.get((r['id'], arm), ('unclaimed',))[0] in ('unclaimed', 'claimed')]
                if not pending: continue
                if not any(attempts.get((r['id'], arm), ('',))[0] == 'claimed' for r in pending):
                    remaining_seconds += 150.  # shard verification and model loading allowance
                for row in pending:
                    cost = estimate(row['suite'], arm)
                    attempt = attempts.get((row['id'], arm))
                    if attempt and attempt[0] == 'claimed':
                        cost = max(0., cost - (now - timestamp(attempt[1])).total_seconds())
                    remaining_seconds += cost
                    suite_finish[row['suite']] = remaining_seconds
        phase_finish[root.name] = remaining_seconds
    for name, summary in stats.items():
        summary['estimated_remaining_hours_until_suite_complete'] = suite_finish.get(name, 0.) / 3600
        summary['estimated_completion_utc'] = (now + dt.timedelta(seconds=suite_finish.get(name, 0.))).isoformat()
        summary['timing_by_arm'] = {}
        for arm in ARMS:
            samples = timings.get((name, arm), [])
            seconds = sum(x[0] for x in samples); tokens = sum(x[1] for x in samples)
            summary['timing_by_arm'][arm] = {'observations': len(samples),
                'mean_seconds': seconds / len(samples) if samples else None,
                'responses_per_inference_hour': len(samples) * 3600 / seconds if seconds else None,
                'generated_tokens_per_second': tokens / seconds if seconds else None,
                'projected_seconds_per_remaining_response': estimate(name, arm)}
    start = json.loads((parent / 'execution.json').read_bytes())['started_at_utc']
    elapsed = (now - timestamp(start)).total_seconds()
    completed = sum(s['completed'] for s in stats.values())
    return {'as_of_utc': now.isoformat(), 'elapsed_hours': elapsed / 3600,
            'completed_responses': completed, 'scheduled_responses': sum(s['scheduled_responses'] for s in stats.values()),
            'overall_responses_per_wall_hour': completed * 3600 / elapsed,
            'last_10_minutes_responses_per_hour': sum(data[-1] for data in loaded) * 6,
            'estimated_remaining_hours': remaining_seconds / 3600,
            'estimated_completion_utc': (now + dt.timedelta(seconds=remaining_seconds)).isoformat(),
            'estimate_caveat': 'Provisional projection, not a confidence interval. Uses observed suite/arm means; unobserved AxonAI uses qualification stress ratio; GPQA uses full-cap GSM8K throughput with a stress floor. Adds 150 seconds per future load. Does not forecast outages, changed response lengths or scoring time.',
            'unobserved_arm_stress_ratio': ratio, 'suites': stats}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.root.resolve()), indent=2))
