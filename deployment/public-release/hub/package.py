"""Prepare the canonical BF16 upload tree and verify it without importing a model."""
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
from pathlib import Path

TOOL = Path(__file__).resolve().parent
NAME = "AxonAI-v1-27B"
PIN = "d34c6389c0ca68df59e11f2a18f3616e7b335066fd8c493acf46044551e1daee"
REV = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MANIFEST = "RELEASE_MANIFEST.json"
SHARDS = [f"model-{n:05}-of-00014.safetensors" for n in range(1,15)]
ORIGINALS = SHARDS + ["model.safetensors.index.json", "config.json", "generation_config.json",
                      "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                      "BASE_LICENSE.Apache-2.0.txt", "BASE_MODEL_CARD.original.md",
                      "standalone_inference.py", "requirements-inference.txt"]
GENERATED = ["README.md","NOTICE.txt","PROVENANCE.json","VALIDATION_SUMMARY.json",
             "SOURCE_ATTRIBUTIONS.rc.json","SOURCE_LICENSE.Apache-2.0.txt"]


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False)+"\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(tree, model, expected_sha):
    """Validate the reviewed allowlist and all bytes, including resolved shard links."""
    manifest = tree / MANIFEST
    require(not manifest.is_symlink() and sha(manifest)==expected_sha, "Manifest digest differs")
    rows = read(manifest)["files"]
    names = [r["name"] for r in rows]
    require(len(names)==len(set(names)) and all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", n) for n in names), "Invalid allowlist")
    require(set(names)|{MANIFEST} == {p.name for p in tree.iterdir() if p.name!=".cache"}, "Unlisted or missing files")
    for row in rows:
        path = tree / row["name"]
        if path.is_symlink():
            require(path.resolve(strict=True)==model.resolve()/path.name, f"Unexpected link for {path.name}")
        require(path.is_file() and path.stat().st_size==row["bytes"] and sha(path)==row["sha256"], f"File differs: {path.name}")
    return names+[MANIFEST]


def header_check(model):
    index = read(model/"model.safetensors.index.json")["weight_map"]
    require(set(index.values())==set(SHARDS) and len(index)==851, "Unexpected BF16 shard index")
    elements = 0
    for name in SHARDS:
        with (model/name).open("rb") as f:
            size = struct.unpack("<Q",f.read(8))[0]
            require(0<size<8_000_000,"Unexpected header length")
            tensors=json.loads(f.read(size))
        tensors.pop("__metadata__",None)
        require(set(tensors)=={k for k,v in index.items() if v==name}, "Header/index mismatch")
        for t in tensors.values():
            require(t["dtype"]=="BF16","Unexpected tensor dtype")
            elements+=(t["data_offsets"][1]-t["data_offsets"][0])//2
    require(elements==26895998464 and read(model/"config.json")["dtype"]=="bfloat16","Unexpected BF16 model")
    return elements


def prepare(tree, model):
    require(sha(model/"SHA256_MANIFEST.json")==PIN, "Unexpected source manifest")
    frozen = {r["relative_path"]:r for r in read(model/"SHA256_MANIFEST.json")["files"]}
    def record(name):
        require(sha(model/name)==frozen[name]["sha256"],"Unexpected source record "+name)
        return read(model/name)
    prov, valid = record("PROVENANCE.json"), record("VALIDATION_SUMMARY.json")
    require(prov["base_revision"]==REV, "Unexpected Qwen revision")
    elements=header_check(model)
    attr=Path("/attribution/SOURCE_ATTRIBUTIONS.rc.json")
    register=read(attr)
    require(register["extracted_training_sha256"]==prov["dataset"]["sha256"] and len(register["sources"])==4230
            and sum(r["training_record_count"] for r in register["sources"])==10684,"Attribution inventory mismatch")
    for name in ORIGINALS:
        src=model/name
        require(not src.is_symlink() and src.stat().st_size==frozen[name]["bytes"], "Unexpected original "+name)
        (tree/name).symlink_to(src)
    shutil.copyfile(TOOL/"bf16"/"README.release.md",tree/"README.md")
    shutil.copyfile(TOOL/"bf16"/"NOTICE.txt",tree/"NOTICE.txt")
    shutil.copyfile(attr,tree/"SOURCE_ATTRIBUTIONS.rc.json")
    shutil.copyfile("/input/SOURCE_LICENSE.txt",tree/"SOURCE_LICENSE.Apache-2.0.txt")
    write(tree/"PROVENANCE.json",dict(model=NAME, precision="BF16", upstream=prov["base_repository_recorded"],
          upstream_revision=REV, original_manifest_sha256=PIN, adapter_sha256=prov["adapter"]["sha256"],
          adapter_config_sha256=prov["adapter_config"]["sha256"], frozen_sft_sha256=prov["dataset"]["sha256"],
          historical_register_sha256=register["historical_register_sha256"], training_articles=4230, training_examples=10684,
          epochs=2, optimizer_steps=668, seed=42, held_out_validation_examples=0,
          merge=prov["merge"]["procedure"], architecture=prov["export_architecture"], parameters=elements,
          source_repository="https://github.com/AXDT-INC/AxonIntelligence", source_commit="1f164f5fe1189758c2687e4c5b6f5f170269c440",
          local_fp16=dict(derived_from=PIN, conversion_receipt_sha256="a1d16eb2cd2d35d1c0aeff623fc9a464fe7ce36a32865c6991149a6840953786",
                          manifest_sha256="c3a6cb1deb5123cc35ccc9c07de36545f82b91a317f97971b6bd56b3403f2e06", included=False)))
    write(tree/"VALIDATION_SUMMARY.json",dict(precision="BF16", independent_load=valid["independent_load"],
          weight_verification=valid["merged_weight_verification"], merge_equations=valid["all_weight_equations_verified"],
          generation_comparison=valid["token_comparison"], smoke=valid["fresh_smoke"], independent_scientific_validation=False,
          tested_inference="2 x L40S 48 GB; BF16 SDPA, 38 GiB placement per GPU, 4096 combined tokens",
          original_validation_sha256=frozen["VALIDATION_SUMMARY.json"]["sha256"]))
    rows=[]
    for name in sorted(ORIGINALS+GENERATED):
        path=tree/name
        rows.append(dict(name=name,bytes=path.stat().st_size,
                         sha256=frozen[name]["sha256"] if name in ORIGINALS else sha(path)))
    write(tree/MANIFEST,dict(model=NAME, precision="BF16", source_manifest_sha256=PIN,
                            self_hash_excluded=True, files=rows))
    (tree/".cache").mkdir()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["prepare","verify"])
    parser.add_argument("model",type=Path)
    args=parser.parse_args()
    stage=Path("/stage")
    dest=stage/NAME
    if args.action=="prepare":
        # A failed build never exposes a partially prepared release or overwrites a reviewed tree.
        with tempfile.TemporaryDirectory(dir=stage) as tmp:
            tree=Path(tmp)/NAME
            tree.mkdir()
            prepare(tree,args.model)
            if dest.exists():
                require((tree/MANIFEST).read_bytes()==(dest/MANIFEST).read_bytes(),
                        "Existing release differs. Use a fresh AXONAI_RELEASE_STAGE to preserve the reviewed candidate.")
            else:
                os.rename(tree,dest)
    else:
        require(sha(args.model/"SHA256_MANIFEST.json")==PIN, "Unexpected source manifest")
        header_check(args.model)
        frozen={r["relative_path"]:r for r in read(args.model/"SHA256_MANIFEST.json")["files"]}
        prepared={r["name"]:r for r in read(dest/MANIFEST)["files"]}
        for name in ORIGINALS:
            require(prepared[name]["sha256"]==frozen[name]["sha256"] and prepared[name]["bytes"]==frozen[name]["bytes"],
                    "Candidate differs from pinned source: "+name)
        names=verify(dest,args.model,sha(TOOL/"bf16"/MANIFEST))
        require(set(names)==set(ORIGINALS+GENERATED+[MANIFEST]),"Unexpected BF16 publication file set")
        print("Verified",len(names),"publication files by streaming all payload hashes.")
    print(NAME,sha(dest/MANIFEST))
