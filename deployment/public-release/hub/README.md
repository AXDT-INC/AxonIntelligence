# Canonical BF16 Hugging Face release candidate

The initial release inventory covers AxonAI-v1-27B in BF16. The FP16 conversion remains local for production V100 inference. The existing conversion receipt, V100 results and earlier FP16 documentation draft are preserved. [Attribution and rights review](../attribution/README.md) · [Local FP16 conversion receipt](../../../models/merged/AxonAI-v1-27B-FP16/.conversion/verification.json) · [V100 validation](../../v100-inference/README.md#production-verification-on-2026-11-25)

## Prepare and verify offline

Run in Bash from the AxonIntelligence repository root on the x86-64 host holding the canonical checkpoint. The Docker build context admits only the Dockerfile and wheel lock. The runtime image uses a digest-pinned Python 3.12 slim base and the project's SHA-256-pinned dependency wheels, including huggingface_hub 1.30.0 and hf_xet 1.8.0. The validated local image ID is `sha256:5c2c7b42210eccef4a4a2dfb35bbd1bb6575219f539c3e1f0dac19ce1b75504f`. Set `AXONAI_RELEASE_IMAGE` to that local ID to select the reviewed build. The review scripts mount read-only at runtime. Build network access is for the pinned wheels. [Dockerfile](Dockerfile) · [Wheel lock](wheels.lock) · [Launch procedure](run.sh)

```bash
export AXONAI_RELEASE_IMAGE=sha256:5c2c7b42210eccef4a4a2dfb35bbd1bb6575219f539c3e1f0dac19ce1b75504f
export AXONAI_RELEASE_STAGE="$PWD/deployment/public-release/outputs/bf16-final"
tool=deployment/public-release/hub/run.sh
"$tool" prepare
"$tool" verify
```

The finalized workflow uses the existing image and the reviewed register. The Dockerfile and wheel lock permit a separate rebuild using `docker build --pull=false -t axonai-hub-cpu:rc1 deployment/public-release/hub`.

For a separate attribution update, `"$tool" fetch` uses PMC EFetch in batches of 35 and ESummary in batches of 100. Requests are serial, with pauses and bounded retries. It caches bibliographic fields and article-level permission metadata. Reconciliation uses cached PMC records and can request Crossref deposits for ambiguous cases. A plain `"$tool" reconcile` replays offline. The accompanying review inputs include article-specific publisher decisions. Unqualified mining URLs alone cannot establish the general article license. A publisher deposit with an unspecified version can corroborate an explicit article attribution grant when its URL matches the PMC annotation. [PMC developer services](https://pmc.ncbi.nlm.nih.gov/tools/developers/) · [Publisher review inputs](publisher-review.json)

Each run uses the ordinary runc runtime, UID/GID 1000:1000, one CPU, a read-only root, dropped capabilities, a bounded temporary filesystem and no GPU devices or Docker socket. Only metadata retrieval and the future upload use a networked runtime. The fetcher receives the historical inventory. Reconciliation additionally receives the frozen receipt. Neither receives Hub credentials. Packaging runs offline with only the BF16 checkpoint mounted read-only. The host's dedicated release output and cache directories must be writable by UID 1000. The defaults are `deployment/public-release/outputs/bf16-final` and `deployment/public-release/.cache`. To stage a revision, use a fresh absolute `AXONAI_RELEASE_STAGE` outside both production model directories. An existing candidate with a different manifest is preserved.

## BF16 file inventory

The current candidate is in `deployment/public-release/outputs/bf16-final/AxonAI-v1-27B`. Earlier candidates remain under `deployment/public-release/outputs` and `deployment/public-release/outputs/focused-rights`. The immutable source files are referenced by symlink to the read-only checkpoint at the same absolute path on host and container. The uploader sends the target bytes. CPU verification checks shard headers against the index, the pinned original manifest, and the digest of every allowlisted file without constructing the model. The publication manifest lists filename, size and SHA-256 for each payload. Its own SHA-256 is recorded separately. The payload is 53,819,669,153 bytes including the manifest, with one copy of each of the 14 shards in the prospective Hub repository.

| Files | Count |
| --- | ---: |
| `model-00001-of-00014.safetensors` through `model-00014-of-00014.safetensors` | 14 |
| `model.safetensors.index.json`, `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | 6 |
| `BASE_LICENSE.Apache-2.0.txt`, `BASE_MODEL_CARD.original.md`, `SOURCE_LICENSE.Apache-2.0.txt`, `NOTICE.txt` | 4 |
| `README.md`, `PROVENANCE.json`, `VALIDATION_SUMMARY.json`, `SOURCE_ATTRIBUTIONS.rc.json`, `RELEASE_MANIFEST.json` | 5 |
| `standalone_inference.py`, `requirements-inference.txt` | 2 |

Final BF16 candidate manifest SHA-256 is `0c82ab39585ad6cd4b31474ff603d23f4591d04c85be749a3c730d92de9a1837`. The fixed allowlist excludes training examples, caches, adapter files and the local FP16 derivative. The package builder copies the original release card as `README.md` to reproduce the reviewed manifest. [Exact file inventory with hashes](bf16/RELEASE_MANIFEST.json) · [Original release card](bf16/README.release.md) · [Package builder](package.py)

The current BF16 model card includes the licensing clarification published on Hugging Face. [Current GitHub card](bf16/README.md#rights) · [Hugging Face revision](https://huggingface.co/AxonDAO/AxonAI-v1-27B/blob/4c27294aa4b00f0765ca361acd43b330a979e32a/README.md)

## Proposed upload after weight release approval

Supply the existing approved Hub model repository, final approved manifest SHA-256 and absolute path to a separately provisioned fine-grained write token file readable by UID 1000. Keep the file outside the checkout and staging/cache directories, with mode 0600. Only its path enters the command. Upload approval must name the final manifest after derivative-license decisions. If approval requires different terms in the payload, regenerate the manifest and obtain authorization for its new digest.

```bash
cd /data/axonai
AXONAI_RELEASE_IMAGE=sha256:5c2c7b42210eccef4a4a2dfb35bbd1bb6575219f539c3e1f0dac19ce1b75504f \
HF_RELEASE_TOKEN_FILE="${HF_RELEASE_TOKEN_FILE:?set the absolute token file path}" \
  deployment/public-release/hub/run.sh upload \
  /data/axonai/deployment/public-release/outputs/bf16-final/AxonAI-v1-27B \
  /data/axonai/models/merged/AxonAI-v1-27B \
  AxonDAO/AxonAI-v1-27B \
  0c82ab39585ad6cd4b31474ff603d23f4591d04c85be749a3c730d92de9a1837
```

An offline check using the pinned Hub SDK selected all 31 candidate files, accepted their sizes and constructed the resume paths on the writable overlay. The reusable uploader checks the manifest against the approved digest, streams hashes, checks the exact set of files and resolves any source links against the supplied model mount. It checks that the destination exists and uses the official `HfApi.upload_large_folder` with explicit filenames, two workers and Xet. The SDK's idempotent repository call uses a private creation fallback. The existing destination retains its visibility. The source and candidate are read-only. A writable overlay at `/candidate/.cache` retains Hub upload progress. The HF cache and resume state are isolated by repository and manifest digest. Repeat the same command after interruption. Network transfer, production service startup and credentialed authentication are outside the offline validation. [Official large-folder upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload#upload-a-large-folder)
