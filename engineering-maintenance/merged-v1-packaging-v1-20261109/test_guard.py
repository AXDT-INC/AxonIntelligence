import unittest,subprocess,sys
from pathlib import Path
class GuardTests(unittest.TestCase):
 def test_metadata_discovery_allowed_import_denied(self):
  script="""import sys,importlib.util
from no_peft_guard import NoPeft
sys.meta_path.insert(0,NoPeft())
assert importlib.util.find_spec('peft') is not None
try:import peft
except ImportError as e:assert 'actual PEFT imports' in str(e)
else:raise AssertionError('PEFT imported')
assert 'peft' not in sys.modules
"""
  r=subprocess.run([sys.executable,'-B','-c',script],cwd=Path(__file__).resolve().parent,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
 def test_transformers_import_with_guard(self):
  script="""import sys
from no_peft_guard import NoPeft
sys.meta_path.insert(0,NoPeft())
from transformers import AutoTokenizer,AutoModelForCausalLM
assert not any(x=='peft' or x.startswith('peft.') for x in sys.modules)
"""
  r=subprocess.run([sys.executable,'-B','-c',script],cwd=Path(__file__).resolve().parent,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
if __name__=='__main__':unittest.main(verbosity=2)
