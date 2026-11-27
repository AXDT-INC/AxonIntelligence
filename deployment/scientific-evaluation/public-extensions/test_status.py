import datetime as dt
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import status


class Progress(unittest.TestCase):
    def test_read_only_counts_privacy_and_sequential_forecast(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); extension = root / 'gpqa-extension'; extension.mkdir()
            now = dt.datetime(2026, 1, 1, 1, tzinfo=dt.timezone.utc)
            def save(path, value): path.write_text(json.dumps(value))
            save(root / 'execution.json', {'started_at_utc': (now-dt.timedelta(seconds=600)).isoformat()})
            save(root / 'qualification.json', {'arms': {
                arm: {'qualified': {'stress_elapsed_s': cost}}
                for arm, cost in [('qwen', 30), ('axonai', 33)]}})
            for phase, name, total in [(root, 'gsm8k', 1), (extension, 'gpqa', 2)]:
                save(phase / 'roster-freeze.json', {'planned': {name: total}, 'eligible': {name: 1},
                                                    'block_arm_order': [['qwen', 'axonai']]})
                rows = [{'id': f'private-synthetic-{name}-{i}', 'suite': name, 'block': 0,
                         'eligible': i == 0, 'gold': 'private-synthetic-key', 'prompt': 'private-synthetic-prompt'}
                        for i in range(total)]
                (phase / 'roster.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with sqlite3.connect(root / 'attempts.sqlite3') as db:
                db.execute('CREATE TABLE attempts(item_id,arm,state,claimed_at,finished_at,outcome)')
                db.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?)', (
                    'private-synthetic-gsm8k-0', 'qwen', 'completed',
                    (now-dt.timedelta(seconds=20)).isoformat(), (now-dt.timedelta(seconds=10)).isoformat(),
                    json.dumps({'status': 'ok', 'text': 'private-synthetic-response', 'stop': 'cap',
                                'elapsed_s': 10, 'output_tokens': 100})))
            before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            report = status.report(root, now)
            self.assertEqual(report['completed_responses'], 1)
            self.assertEqual(report['scheduled_responses'], 4)
            self.assertEqual(report['last_10_minutes_responses_per_hour'], 6)
            self.assertEqual(report['suites']['gsm8k']['format_failures'], 1)
            self.assertEqual(report['suites']['gsm8k']['inference_failures'], 0)
            self.assertEqual(report['suites']['gsm8k']['cap_hits'], 1)
            # Remaining GSM8K AxonAI: load 150 + 11. GPQA: 150+30+150+33.
            self.assertAlmostEqual(report['estimated_remaining_hours'] * 3600, 524)
            self.assertNotIn('private-synthetic', json.dumps(report))
            self.assertFalse((extension / 'attempts.sqlite3').exists())
            self.assertEqual(before, {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()})


if __name__ == '__main__': unittest.main()
