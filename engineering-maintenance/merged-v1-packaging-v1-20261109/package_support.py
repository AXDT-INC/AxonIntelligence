import json,shutil,sys,platform,importlib.metadata,datetime
from pathlib import Path
from runtime_safeguards import write,binding
P=Path(__file__).resolve().parent;ROOT=Path('/data/axonai');M=ROOT/'models/merged/AxonAI-v1-27B';BASE=ROOT/'models/base/Qwen3.8-27B';release=ROOT/'engineering-maintenance/release-window-preparation-v1-20261103'
for src,name in [(P/'standalone_inference.py','standalone_inference.py'),(P/'verify_manifest.py','verify_manifest.py'),(BASE/'LICENSE','BASE_LICENSE.Apache-2.0.txt'),(BASE/'README.md','BASE_MODEL_CARD.original.md'),(release/'release-repository/ATTRIBUTIONS.DRAFT.json','SOURCE_ATTRIBUTIONS.inherited-draft.json'),(release/'RELEASE_RIGHTS_STATUS.json','RELEASE_RIGHTS_STATUS.inherited.json')]:
 assert not (M/name).exists();shutil.copyfile(src,M/name)
versions={n:importlib.metadata.version(n) for n in ['torch','transformers','accelerate','safetensors','tokenizers','huggingface-hub','peft','numpy']}
write(M/'DEPENDENCIES.json',{'python':sys.version,'platform':platform.platform(),'versions':versions,'peft_required_for_inference':False,'peft_used_only_to_create_package':True,'cuda_runtime':__import__('torch').version.cuda,'note':'Exact installed versions recorded; this package does not bundle dependency wheels or guarantee their availability from another server package index.'})
(M/'requirements-inference.txt').write_text('\n'.join(f'{n}=={versions[n]}' for n in ['torch','transformers','accelerate','safetensors','tokenizers','huggingface-hub'])+'\n')
write(M/'PROVENANCE.json',{'name':'AxonAI-v1-27B','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'base_repository_recorded':'Qwen/Qwen3.8-27B','base_revision':'1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0','base_architecture_class':'Qwen3_5ForConditionalGeneration','export_architecture':'Qwen3_5ForCausalLM','export_scope':'Text-only AutoModelForCausalLM path, matching validated v1 inference; no vision encoder or image/video support','adapter':binding(ROOT/'engineering-maintenance/main-sft-run-20261029T150039Z/final_adapter/adapter_model.safetensors'),'adapter_config':binding(ROOT/'engineering-maintenance/main-sft-run-20261029T150039Z/final_adapter/adapter_config.json'),'v11_used':False,'training':'Inherited completed v1: 668 optimizer steps, two epochs; no new training','merge':json.loads((P/'MERGE_RESULT.json').read_text()),'base_files':json.loads((P/'PREFLIGHT.json').read_text())['verified_source_files'],'dataset':binding(ROOT/'engineering-maintenance/frozen-v1-training-extraction-v1-20281029/train-10684.jsonl'),'knowledge_limitations':'Primarily biomedical SFT; source-context deficiencies remain; no independent biomedical superiority established. RAG remains separate.','distribution':'Internal local packaging authorized; no public distribution or deployment performed. Prior release-rights decisions remain separate.'})
(M/'NOTICE.txt').write_text('''AxonAI-v1 merged research model, local modification dated November 9, 2026.
Base attribution: Qwen team, pinned Qwen/Qwen3.8-27B revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0. See the retained upstream model card and Apache-2.0 licence.
Modification: the AxonAI-v1 biomedical LoRA adapter was merged into the base text model using FP32 arithmetic and serialized as BF16. This is a text-only export; the original base vision components are not included. Tokenizer/configuration are locally self-contained.
AxonAI-v1 adapter authorship/organization attribution and inherited source attribution records are retained in PROVENANCE.json and SOURCE_ATTRIBUTIONS.inherited-draft.json. The source register is inherited evidence, not a new legal approval or a grant to distribute source articles.
No Qwen, source-author or scientific-reviewer endorsement is implied. Original base and adapter were not modified. No experimental v1.1 weights were used.
The included base Apache-2.0 licence does not select or settle a licence for the combined derivative. The prior adapter/public-release licence determination remains pending; internal packaging does not authorize publication. See RELEASE_RIGHTS_STATUS.inherited.json.
''')
(M/'README.md').write_text('''# AxonAI-v1-27B — standalone merged research model

This directory is a self-contained **text-only BF16** export of the pinned Qwen base plus the completed biomedical AxonAI-v1 LoRA adapter. It requires neither the original base directory nor PEFT/adapter files at inference. It does not include RAG or the experimental v1.1 adapter.

The stored architecture is `Qwen3_5ForCausalLM` / `qwen3_5_text`, as used by the validated text inference path; the historical repository/path label is Qwen3.8-27B. Use the actual saved configuration. This export does not contain the multimodal vision tower and does not claim image/video support.

Fourteen Safetensors shards contain 26,895,998,464 BF16 parameters (approximately 53.79 GB of tensor data). Merge arithmetic was FP32, followed by one BF16 rounding. Separate LoRA and merged matrix operations can yield different logits or generated tokens. See VALIDATION_SUMMARY.json for measured comparisons; bitwise equivalence is not assumed.

## Load on another server

1. Transfer this complete directory, including the index, all shards, tokenizer, chat template, configuration, provenance and notices. Run the SHA-256 verifier before loading.
2. Use a separate Python environment matching DEPENDENCIES.json. Obtain compatible wheels for the exact versions in requirements-inference.txt, including the PyTorch CUDA build, from a trusted source appropriate to that server. The dependency wheels are not bundled; alternate versions/backends require revalidation.
3. The tested GPU layout is **two NVIDIA L40S 48 GB cards**, BF16, SDPA, no CPU/disk offload, 38 GiB weight-placement budget per card. Use a driver compatible with the recorded PyTorch CUDA runtime. Other GPU models, single-GPU layouts, quantization and serving engines are unvalidated.
4. Allow at least 64 GB host RAM; 128 GB is a conservative operational recommendation, not a measured minimum. Reserve 120 GB free disk if retaining an incoming copy plus the model. The creation procedure required substantially more CPU memory than inference; it need not be repeated.
5. Use the validated 4,096-token input-plus-output envelope. The example rejects oversized requests rather than silently truncating. Long-context capability implied by the upstream configuration has not been validated here.

```bash
python3 verify_manifest.py --root . --manifest SHA256_MANIFEST.json
python3 -m venv .venv
. .venv/bin/activate
# Install the pinned dependencies from the appropriate trusted wheel source.
python -m pip install -r requirements-inference.txt
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python standalone_inference.py --model . \
  --prompt 'Explain why technical replicates do not increase the number of independent biological samples.'
```

The local example uses `AutoModelForCausalLM.from_pretrained(..., local_files_only=True, trust_remote_code=False)`. No PEFT package is needed for inference. No model-serving endpoint is started.

## Research limitations

This is an experimental biomedical-focused SFT model, not a clinically validated system. Prior base/adapter comparisons did not establish biomedical superiority. Missing source context, unsupported specificity, citation omissions and reasoning failures remain possible. Merging changes packaging; it does not repair knowledge, grounding or safety deficiencies. Do not treat generated statements as clinical advice or independently verified scientific conclusions.

A future retrieval service may insert evidence into user messages using this tokenizer's chat template. The model can run without retrieval, but source-specific questions should supply appropriate evidence. Retrieval indexing, freshness, permissions and citation checks remain separate services.

## Rights and status

Local internal packaging and validation were authorized. Publication, deployment and a public derivative licence were not authorized. The base licence and original notices are preserved; inherited release-rights status remains pending where previously recorded. No new rights investigation or claim of clearance was made. Keep provenance and attribution notices with any authorized copy.
''')
print('SUPPORT_FILES_READY')
