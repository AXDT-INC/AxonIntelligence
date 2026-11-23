#!/usr/bin/env python3
"""Unexecuted candidate smoke runner. Authorization checks precede ALL ML imports."""
import argparse
from datetime import datetime, timezone
import importlib.abc
import importlib.metadata
import importlib.util
import os
import re
import sys
from audit_io import Directory, OUTPUT_MANIFEST, checked_manifest, demand, emit_receipt, parse_json

BLOCKED = {'triton', 'flash_attn', 'fla', 'causal_conv1d', 'kernels'}

class NoFusedImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in BLOCKED:
            raise ImportError('GPU kernel/JIT import prohibited by reviewed initial policy: '+fullname)
        return None

def authorization(path, explicit, now=None):
    demand(explicit, 'separate explicit GPU authorization required')
    parent, name = os.path.split(path)
    with Directory(parent) as d:
        auth = parse_json(d.read(name))
    demand(auth.get('purpose') == 'AxonAI-v1 independent FP16 validation' and auth.get('authorized') is True,
           'invalid authorization purpose')
    devices = auth.get('exclusive_gpu_uuids')
    demand(isinstance(devices, list) and len(devices) == 2 and len(set(devices)) == 2,
           'two exclusive UUIDs required')
    demand(all(isinstance(x,str) and re.fullmatch(r'GPU-[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}', x)
               for x in devices), 'invalid GPU UUID')
    expiry = datetime.fromisoformat(auth.get('expires_utc', '').replace('Z', '+00:00'))
    demand(expiry.tzinfo is not None and expiry > (now or datetime.now(timezone.utc)), 'authorization expired')
    demand(auth.get('scheduler_reservation_confirmed') is True, 'scheduler reservation confirmation required')
    demand(auth.get('runtime_audit_passed') is True, 'separate runtime audit required')
    demand(re.fullmatch('[0-9a-f]{64}', str(auth.get('verified_output_manifest_sha256',''))), 'verified artifact anchor required')
    demand(re.fullmatch('sha256:[0-9a-f]{64}', str(auth.get('runtime_image_digest',''))), 'verified image digest required')
    demand(os.environ.get('AXONAI_RUNTIME_IMAGE_DIGEST') == auth['runtime_image_digest'], 'image identity declaration mismatch')
    demand(os.environ.get('NVIDIA_VISIBLE_DEVICES') == ','.join(devices), 'visible UUID declaration mismatch')
    # These are operator attestations, not a live scheduler query or cryptographic grant.
    return auth

def execute(args):
    auth = authorization(args.authorization, args.authorize_gpu_inference)
    demand(sys.version_info[:2] == (3,12), 'Python 3.12 required')
    required = {'torch':'2.13.0+cu126', 'transformers':'5.16.1', 'accelerate':'1.14.0',
                'safetensors':'0.8.0', 'tokenizers':'0.23.2', 'huggingface-hub':'1.30.0'}
    for package, version in required.items():
        demand(importlib.metadata.version(package) == version, 'candidate dependency mismatch: '+package)
    for name in BLOCKED-{'triton'}:
        demand(importlib.util.find_spec(name) is None, 'optional fused-kernel package must be absent: '+name)
    demand(not (BLOCKED & set(sys.modules)), 'blocked kernel module already imported')
    sys.meta_path.insert(0, NoFusedImports())
    for name,value in {'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1',
                       'TORCH_COMPILE_DISABLE':'1', 'TORCHDYNAMO_DISABLE':'1'}.items():
        os.environ[name] = value
    with Directory(args.model) as model_dir:
        checked_manifest(model_dir, OUTPUT_MANIFEST, auth['verified_output_manifest_sha256'])
        config = parse_json(model_dir.read('config.json'))
        demand(config.get('dtype') == 'float16' and config.get('mamba_ssm_dtype') == 'float32', 'invalid runtime dtype policy')
    # FIRST ML import / potential CUDA contact: strictly after the gate above.
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    demand(torch.version.cuda == '12.6', 'CUDA 12.6 build required')
    demand('sm_70' in torch.cuda.get_arch_list(), 'binary lacks sm_70')
    demand(torch.cuda.device_count() == 2, 'exactly two GPUs must be visible')
    for i, uuid in enumerate(auth['exclusive_gpu_uuids']):
        properties = torch.cuda.get_device_properties(i)
        demand((properties.major, properties.minor) == (7,0), 'non-V100 compute capability')
        reported = str(getattr(properties, 'uuid', ''))
        demand(reported.lower().removeprefix('gpu-') == uuid.lower().removeprefix('gpu-'), 'GPU UUID not verified')
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16,
            attn_implementation='sdpa', use_kernels=False, local_files_only=True, trust_remote_code=False,
            device_map='balanced', max_memory={0:'27GiB', 1:'27GiB'})
    demand(all(str(v) in {'0','1','cuda:0','cuda:1'} for v in model.hf_device_map.values()), 'offload outside initial policy')
    model.eval().requires_grad_(False)
    # Do NOT call model.half(): intentional architecture-specific FP32 ops remain intact.
    text = tokenizer.apply_chat_template([{'role':'user','content':args.prompt}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, add_special_tokens=False, return_tensors='pt')
    demand(inputs['input_ids'].shape[-1] <= 256, 'initial input budget exceeded')
    inputs = {k:v.to(model.get_input_embeddings().weight.device) for k,v in inputs.items()}
    with torch.inference_mode(), sdpa_kernel(backends=[SDPBackend.MATH]):
        result = model.generate(**inputs, generation_config=GenerationConfig(do_sample=False,
                max_new_tokens=32, use_cache=True, eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id), return_dict_in_generate=True, output_scores=True)
        demand(all(torch.isfinite(x).all().item() for x in result.scores), 'nonfinite generation scores')
    emit_receipt({'status':'SMOKE_ONLY', 'text':tokenizer.decode(result.sequences[0], skip_special_tokens=True),
                  'frozen_seven_cases_passed':False, 'inference_equivalence_established':False})

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--authorize-gpu-inference', action='store_true')
    p.add_argument('--authorization', default='/authorization/gpu-reservation.json')
    p.add_argument('--model', default='/model')
    p.add_argument('--prompt', default='Return the sum of 5 and 7 as JSON.')
    a = p.parse_args()
    try:
        execute(a)
        return 0
    except Exception as exc:
        emit_receipt({'status':'FAIL', 'phase':'candidate-inference', 'error':str(exc)})
        return 1

if __name__ == '__main__':
    sys.exit(main())
