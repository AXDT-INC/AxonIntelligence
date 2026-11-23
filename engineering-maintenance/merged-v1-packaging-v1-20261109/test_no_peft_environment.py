import unittest,subprocess,sys
from pathlib import Path
PREFIX="""import sys,importlib.util
from pathlib import Path
s=importlib.util.spec_from_file_location('guard',Path('no_peft_guard.v2.py'));g=importlib.util.module_from_spec(s);s.loader.exec_module(g);g.install()
"""
class NoPeftEnvironment(unittest.TestCase):
 def run_code(self,code):
  r=subprocess.run([sys.executable,'-B','-c',PREFIX+code],cwd=Path(__file__).resolve().parent,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
 def test_package_reported_absent_and_import_blocked(self):
  self.run_code("""assert importlib.util.find_spec('peft') is None
try:import peft
except ImportError:pass
else:raise AssertionError('PEFT imported')
assert 'peft' not in sys.modules
""")
 def test_actual_model_class_import_without_peft(self):
  self.run_code("""from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from transformers.utils import is_peft_available
assert not is_peft_available()
assert not any(x=='peft' or x.startswith('peft.') for x in sys.modules)
""")
if __name__=='__main__':unittest.main(verbosity=2)
