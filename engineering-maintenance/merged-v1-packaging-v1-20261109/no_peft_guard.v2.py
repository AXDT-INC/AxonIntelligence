"""Simulate PEFT being uninstalled while also blocking direct PEFT imports."""
import importlib.abc,importlib.machinery,importlib.util,sys
class DenyPeftLoader(importlib.abc.Loader):
 def create_module(self,spec):return None
 def exec_module(self,module):raise ImportError('Standalone validation forbids actual PEFT imports')
class NoPeft(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='peft' or fullname.startswith('peft.'):
   return importlib.machinery.ModuleSpec(fullname,DenyPeftLoader(),is_package=fullname=='peft')
def install():
 assert not any(n=='peft' or n.startswith('peft.') for n in sys.modules)
 original=importlib.util.find_spec
 def unavailable(name,package=None):
  if name=='peft' or name.startswith('peft.'):return None
  return original(name,package)
 importlib.util.find_spec=unavailable
 sys.meta_path.insert(0,NoPeft())
