import unittest
from score_cases import score

GOOD = dict(sum='{"sum":12,"status":"ok"}',ratio='{"ratio":3}',units='{"mg":250}',
            replicates='1',insufficient='NO',citation='{"source":"N2","watts":4}',
            ordering='[0.02,0.12,1.2]')


class Scoring(unittest.TestCase):
    def test_frozen_controls(self):
        self.assertEqual(score(GOOD),dict.fromkeys(GOOD,True))

    def test_invalid_citations_and_numbers(self):
        for value in ('{"source":"N3","watts":4}', '{"source":"N2","watts":4,"watts":4}',
                      '{"source":"N2","watts":true}', '{"source":"N2","watts":NaN}'):
            self.assertFalse(score(dict(GOOD,citation=value))['citation'])

    def test_output_must_be_continuation_only(self):
        self.assertFalse(score(dict(GOOD,ratio='Prompt preceding {"ratio":3}'))['ratio'])
        with self.assertRaises(ValueError):
            score({'sum':GOOD['sum']})
