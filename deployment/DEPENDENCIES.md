# Candidate runtime dependencies: unresolved offline

No network access, wheel acquisition, installation, image inspection/execution,
or dependency resolver run against real artifacts occurred during preparation.
The baseline model's `DEPENDENCIES.json` records the old cu132 environment, not
a complete dependency graph for this cu126 candidate. The complete transitive
closure **cannot be established from the files available locally**. This is an
explicit unresolved deliverable constraint, not a claim of a complete lock.

| Direct requirement | Candidate pin | Verification state |
|---|---|---|
| Python | 3.12 (patch/build unresolved) | No candidate interpreter/image tested |
| PyTorch | 2.13.0+cu126 | Wheel bytes/hash and sm_70 build not locally verified |
| Transformers | 5.16.1 | Source-version baseline only; wheel/dependencies unverified |
| Accelerate | 1.14.0 | Wheel/dependencies unverified |
| Safetensors | 0.8.0 | Wheel/dependencies unverified; not installed for CPU tests |
| Tokenizers | 0.23.2 | Wheel/dependencies unverified |
| Hugging Face Hub | 1.30.0 | Wheel/dependencies unverified |

`dependency-audit.json` records an advisory checklist of potentially required
transitives, all unresolved. It is not an exhaustive graph and intentionally has
no invented versions or hashes. CUDA native component versions, Torch's Triton
package pin, NumPy and all conditional/extras dependencies must come from the
actual wheel METADATA. A Triton distribution may be a required Torch dependency;
its presence would not authorize Triton JIT. Initial inference blocks its imports
and must fail closed if that is incompatible with the candidate Torch build.
No PEFT, FlashAttention, FLA, causal-conv1d, Hub `kernels`, torchvision or torchaudio
is requested by the initial text-only runtime.

After separate approval to acquire artifacts, supply a dedicated wheelhouse with
one verified wheel per distribution. `audit_wheelhouse.py` can then inspect every
METADATA `Requires-Dist` entry recursively, resolve conditional dependencies for
CPython 3.12/Linux x86_64, check extras/version constraints and operator-declared
wheel tags, hash every wheel, reject missing/extra/duplicate distributions, and
produce a sorted `requirements.lock` plus `dependency-audit.resolved.json`.
It does not install, extract, execute or import wheel code and has no networking.
The current marker target is Python 3.12.0; review the marker environment against
the exact chosen image's patch/build before using it. All package metadata fields
are retained, including inactive platform/extra dependency branches. The audit
itself requires `packaging`, already installed on this host; its test version is
recorded in `test-results.json`.

Example form ONLY after artifacts, tags and output directory are approved:

```text
python3 -B audit_wheelhouse.py --wheelhouse /approved/wheelhouse \
  --accepted-tag <each-approved-cp312/abi3/pure-wheel-tag> \
  --output-directory /approved/new-lock-directory
```

The tool refuses network/URL dependencies and existing lock outputs. A PASS
metadata-closure result is not a security review of wheels, native ABI validation,
or a reproducible image claim. Wheel origin/authenticity, signatures if supplied,
digests, Python patch, base-image digest, glibc/libstdc++/libgcc and other system
libraries, CUDA 12.6/cuDNN/cuBLAS/NCCL component versions, sm_70 binary coverage,
driver compatibility, and Torch/Transformers imports must still be audited.

`Dockerfile.cpu` needs a resolved Python image digest but installs nothing.
`Dockerfile.v100` deliberately requires unavailable `wheelhouse/`,
`requirements.lock`, and an explicit `BASE_IMAGE`; it has no default floating
base. It installs only offline binary wheels with required hashes. It has not
been built or syntax-checked with Docker. The Compose image references are
UNBUILT placeholders, not pinned or verified images. Future launch must substitute
verified digest references and preserve `pull_policy: never`.

The preflight identified R580.173.02 on V100 sm_70 and a plausible cu126 route.
This package does not reconfirm hardware, call NVIDIA tools, or certify that route.
The candidate native PyTorch DeltaNet fallback and SDPA math path remain
experimental until separately authorized validation. FP16 weight finiteness is
necessary but not sufficient: recurrence/norm overflow, optional imports, kernel
availability, device-map memory and numerical output quality remain open.
