import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from public_suite import number, parse_answer, render, paired_stats
from benchmark_run import Ledger


class Scoring(unittest.TestCase):
    def test_labels_and_single_final_contract(self):
        for value in ('yes','no','maybe'):
            self.assertEqual(parse_answer('FINAL '+value.upper(),'pubmedqa'),value)
        for value in 'ABCD':self.assertEqual(parse_answer('FINAL '+value,'medmcqa'),value.lower())
        for bad in ('yes','FINAL yes\nexplanation','FINAL no\nFINAL yes','FINAL maybe.','FINAL yes or no',''):
            self.assertIsNone(parse_answer(bad,'pubmedqa'))
        self.assertEqual(parse_answer('Working\nFINAL B','medmcqa'),'b')

    def test_numbers(self):
        for raw,want in [('1,234.00','1234'),('-1,234.50','-1234.5'),('+42','42'),('-0.0','0')]:
            self.assertEqual(number(raw),want)
        for bad in ('1,23','1,2345','1e3','NaN','inf','12 dollars','1/2'):
            self.assertIsNone(number(bad))
        self.assertEqual(parse_answer('Two plus two\nFINAL 4','gsm8k'),'4')

    def test_render_options_and_context(self):
        text=render('medmcqa','synthetic question',options=['w','x','y','z'])
        for i,letter in enumerate('ABCD'):self.assertIn(f'{letter}. '+['w','x','y','z'][i],text)
        self.assertIn('Context:\none\ntwo',render('pubmedqa','synthetic?',context=['one','two']))
        self.assertNotIn('\r',render('gsm8k','one\r\ntwo'))

    def test_paired_direction_and_exact_test(self):
        result=paired_stats([(0,1)]*6)
        self.assertEqual(result['axonai_wins'],6)
        self.assertEqual(result['difference_percentage_points'],100)
        self.assertAlmostEqual(result['exact_mcnemar_two_sided_p'],.03125)
        self.assertEqual(paired_stats([(1,1),(0,0)])['exact_mcnemar_two_sided_p'],1)


class Restart(unittest.TestCase):
    def test_interrupted_claim_is_never_reissued(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);row={'id':'fixture','block':0};job={'id':'fixture','prompt':'synthetic','max_new_tokens':4}
            ledger=Ledger(root,{'fixture':1});self.assertTrue(ledger.claim(row,'qwen',job))
            ledger.db.close();resumed=Ledger(root,{'fixture':1})
            self.assertFalse(resumed.claim(row,'qwen',job))
            self.assertEqual(resumed.counts(),{'missing_interrupted':1})
            self.assertTrue(resumed.claim(row,'axonai',job));resumed.finish('fixture','axonai',{'status':'ok'})
            with self.assertRaises(RuntimeError):resumed.finish('fixture','axonai',{'status':'ok'})
            resumed.db.close()
            with self.assertRaises(RuntimeError):Ledger(root,{'fixture':2})


if __name__=='__main__':unittest.main()
