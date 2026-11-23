#!/usr/bin/env python3
import argparse,json,hashlib
from pathlib import Path

def verify(root,manifest):
 root=Path(root).resolve();m=json.loads(Path(manifest).read_text());total=0
 for item in m['files']:
  r=Path(item['relative_path']);assert not r.is_absolute() and '..' not in r.parts
  f=(root/r).resolve();assert f.is_relative_to(root) and f.is_file();assert f.stat().st_size==item['bytes'],str(f)
  with f.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==item['sha256'],str(f)
  total+=item['bytes']
 return {'all_passed':True,'files':len(m['files']),'bytes':total,'root':str(root)}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--manifest',required=True);a=p.parse_args();print(json.dumps(verify(a.root,a.manifest),indent=2))
