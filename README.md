# AxonAI-v1

AxonAI-v1 is an experimental, text-only 27B language model fine-tuned on scientific instruction and response examples derived from biomedical literature. This repository contains the training and packaging code, reproducibility records, and a private inference service. The repository includes selected model metadata. Distribution of model weights, adapters or training examples requires separate approval.

Production validation used the merged FP16 model on two 32 GB V100 GPUs. Three independent requests through a private Unix socket produced the expected answers, including a 255-token response with a 256-token cap. These tests establish that the service loads the model and serves consecutive requests. Scientific performance relative to the base model and clinical suitability remain unestablished. [Validation results](deployment/v100-inference/README.md#production-verification-on-2026-11-25)

## Quick start

From the repository root, verify the tracked training, merge and conversion records on CPU. This check uses Python's standard library and requires neither model weights nor a service connection.

```sh
python3 -B deployment/reproducibility/verify_records.py
```

If you have access to a running instance, set `AXONAI_SOCKET` to its Unix socket **as seen by the client**. Check readiness and submit one independent request.

```sh
curl --silent --show-error --fail --unix-socket "${AXONAI_SOCKET:?set the client-visible socket path}" \
  http://localhost/ready
curl --silent --show-error --fail --max-time 150 --unix-socket "${AXONAI_SOCKET:?}" \
  -H 'Content-Type: application/json' --data '{"prompt":"What is a technical replicate?","max_new_tokens":64}' \
  http://localhost/generate
```

A successful generation returns `text`, `input_tokens` and `output_tokens`. The FP16 service accepts at most 2,048 tokens after chat formatting and generates up to 256 tokens per request. Operators with the pinned checkpoint and two reserved GPUs can use the [standalone service instructions](deployment/v100-inference/README.md#start-or-restore-a-private-instance).

## Release and attribution

AxonDAO licenses its first-party source code and documentation under Apache-2.0. Qwen assets retain their upstream license and notices. Third-party article materials retain their applicable Creative Commons terms and exceptions. The historical attribution register documents provenance and contains known errors. Redistribution clearance for model weights, adapters and training datasets remains pending. [License text](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md) · [Release scope, lineage and limitations](RELEASE.md)
