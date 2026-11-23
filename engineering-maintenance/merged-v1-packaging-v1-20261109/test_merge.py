import unittest,tempfile,json,torch
from pathlib import Path
from peft import LoraConfig,get_peft_model
class MergeTests(unittest.TestCase):
 def test_fp32_merge_then_round(self):
  torch.manual_seed(42);base=torch.nn.Sequential(torch.nn.Linear(8,8,bias=False)).to(torch.bfloat16);original=base[0].weight.float().clone();m=get_peft_model(base.float(),LoraConfig(r=2,lora_alpha=4,target_modules=['0']))
  with torch.no_grad():m.base_model.model[0].lora_B['default'].weight.fill_(.003)
  layer=m.base_model.model[0];expected=original+layer.lora_B['default'].weight@layer.lora_A['default'].weight*2
  out=m.merge_and_unload(safe_merge=True);self.assertTrue(torch.equal(out[0].weight,expected));out.bfloat16();self.assertTrue(torch.equal(out[0].weight,expected.bfloat16()));self.assertFalse(any('lora' in n for n,_ in out.named_parameters()))
 def test_safe_merge_rejects_nan(self):
  base=torch.nn.Sequential(torch.nn.Linear(8,8,bias=False));m=get_peft_model(base,LoraConfig(r=2,target_modules=['0']))
  with torch.no_grad():m.base_model.model[0].lora_B['default'].weight.fill_(float('nan'))
  with self.assertRaises(ValueError):m.merge_and_unload(safe_merge=True)
 def test_frozen_prompts_no_contaminated(self):
  p=Path(__file__).resolve().parent;v=json.loads((p/'VALIDATION_PROMPTS.json').read_text());self.assertEqual(len(v['prompts']),6);self.assertEqual(len({x['id'] for x in v['prompts']}),6);self.assertTrue(all(x['kind']!='CONTAMINATED_SOURCE_DIAGNOSTIC_ONLY' for x in v['prompts']))
 def test_worker_uses_v1_only(self):
  p=Path(__file__).resolve().parent
  self.assertNotIn('mixed-scope-continuation', (p/'merge_cpu.py').read_text());self.assertIn('main-sft-run-20261029T150039Z', (p/'merge_cpu.py').read_text())
if __name__=='__main__':unittest.main(verbosity=2)
