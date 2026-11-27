import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import gpqa


class Extension(unittest.TestCase):
    def test_deterministic_private_choice_permutation(self):
        labels=set()
        for i in range(40):
            row={'Record ID':str(i),'Question':'Synthetic comparison question?',
                 'Correct Answer':'synthetic correct','Incorrect Answer 1':'synthetic wrong one',
                 'Incorrect Answer 2':'synthetic wrong two','Incorrect Answer 3':'synthetic wrong three',
                 'High-level domain':'synthetic'}
            out=gpqa.project(row);order=out['choice_order'];labels.add(out['gold'])
            self.assertEqual(order,sorted(range(4),key=lambda j:(hashlib.sha256(f"42/gpqa-{i}/{j}".encode()).hexdigest(),j)))
            self.assertEqual(out['gold'],'abcd'[order.index(0)])
            self.assertIn(out['gold'].upper()+'. synthetic correct',out['prompt'])
            self.assertNotIn('Correct Answer',out['prompt'])
            self.assertNotIn('High-level domain',out['prompt'])
        self.assertEqual(labels,set('abcd'))

    def test_unicode_line_separators_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'roster.jsonl'
            row={'prompt':'synthetic\u2028line\u2029paragraph\x85separator'}
            gpqa.write_roster(path,[row])
            lines=path.read_text().splitlines()
            self.assertEqual(len(lines),1)
            self.assertEqual(json.loads(lines[0]),row)

    def test_parent_progress_verifier_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'roster.jsonl').write_text('synthetic roster\n')
            ledger=gpqa.runner.Ledger(root,{'synthetic':1})
            row={'id':'synthetic-1','block':0};job={'id':row['id'],'prompt':'synthetic','max_new_tokens':4}
            ledger.claim(row,'qwen',job);ledger.finish(row['id'],'qwen',{'status':'ok','text':'FINAL A'})
            before=gpqa.parent_progress(root)
            row2={'id':'synthetic-2','block':0};ledger.claim(row2,'qwen',dict(job,id=row2['id']))
            after=gpqa.verify_parent_progress(root,before)
            self.assertEqual(after['states'],{'claimed':1,'completed':1})
            with ledger.db:ledger.db.execute("UPDATE attempts SET outcome='{}' WHERE state='completed'")
            with self.assertRaises(AssertionError):gpqa.verify_parent_progress(root,before)
            ledger.db.close()

    def test_extension_analysis_and_additive_combination(self):
        with tempfile.TemporaryDirectory() as directory:
            parent=Path(directory);root=parent/'gpqa-extension';root.mkdir()
            (parent/'roster.jsonl').write_text('unchanged parent roster')
            original={'suites':{'pubmedqa':{'eligible':500}},'primary':'pubmedqa'}
            gpqa.save(parent/'results.json',original);gpqa.save(parent/'completion.json',{'all_slots_attempted':True})
            original_bytes=(parent/'results.json').read_bytes()
            row={'id':'gpqa-synthetic','suite':'gpqa','gold':'c','eligible':True,'block':0,'subject':'synthetic'}
            (root/'roster.jsonl').write_text(json.dumps(row)+'\n');ledger=gpqa.runner.Ledger(root,{'synthetic':2})
            for arm,answer in [('qwen','A'),('axonai','C')]:
                ledger.claim(row,arm,{'id':row['id'],'prompt':'synthetic','max_new_tokens':4})
                ledger.finish(row['id'],arm,{'status':'ok','text':'FINAL '+answer,'elapsed_s':1.,'output_tokens':3,'stop':'eos','gpu':[]})
            ledger.db.close();result=gpqa.analyze_extension(root)
            self.assertEqual(result['gpqa'],'included')
            self.assertIsNone(result['primary'])
            self.assertEqual(result['suites']['gpqa']['paired']['axonai_wins'],1)
            self.assertNotIn('exact_mcnemar_two_sided_p',result['suites']['gpqa']['paired'])
            gpqa.save(root/'completion.json',{'all_slots_attempted':True});gpqa.combine(parent,root)
            combined=json.loads((parent/'combined-results.json').read_bytes())
            self.assertEqual(set(combined['suites']),{'pubmedqa','gpqa'})
            self.assertEqual((parent/'results.json').read_bytes(),original_bytes)


if __name__=='__main__':unittest.main()
