"""Resume an approved manifest to an existing approved Hub model repo."""
import argparse
from pathlib import Path
from huggingface_hub import HfApi
from package import verify

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("repo_id")
    p.add_argument("approved_manifest_sha256")
    p.add_argument("model_mount",type=Path)
    args=p.parse_args()
    tree=Path("/candidate")
    files=verify(tree,args.model_mount,args.approved_manifest_sha256)
    api=HfApi(token=Path("/run/secrets/hf_token").read_text().strip())
    api.repo_info(args.repo_id,repo_type="model")
    # The SDK calls create_repo(exist_ok=True). Keep its creation fallback private.
    api.upload_large_folder(repo_id=args.repo_id, repo_type="model", folder_path=tree,
                            private=True, allow_patterns=files, num_workers=2)
