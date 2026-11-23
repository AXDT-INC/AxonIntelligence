"""Allow package discovery, but prohibit any actual PEFT import."""
import importlib.abc,importlib.machinery
class DenyPeftLoader(importlib.abc.Loader):
 def create_module(self,spec):return None
 def exec_module(self,module):raise ImportError('Standalone validation forbids actual PEFT imports')
class NoPeft(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='peft' or fullname.startswith('peft.'):
   return importlib.machinery.ModuleSpec(fullname,DenyPeftLoader(),is_package=fullname=='peft')
