# V2 targeted safety correction — awaiting independent review

The prior audit's NO-GO finding was that a post-verification receipt could be
written inside the authoritative source or derived artifact. Prior sources,
manifest, tests and reports are preserved byte-for-byte in `audit-history/v1/`.
No v1 numerical conversion function was optimized or changed: AST comparisons
confirmed unchanged cast, integer oracle and statistical calculations.

## Receipt policy

Both production CLIs require `--receipt` and construct `ReceiptDestination`
before any model processing. Only a single filename directly under
`/data/axonai/receipts/` is accepted. The operator has created this directory as
UID/GID 1000, mode 0700; it remained empty throughout preparation/testing.
No sudo, automatic directory creation or alternate location is used.

Relative/noncanonical paths, traversal, symlinks in path components, existing
files/directories/FIFOs/symlinks, wrong permissions and mountpoint aliases are
rejected. The root must be private (0700), owned by the effective user, and
writable/searchable. The validated directory descriptor remains open through the
operation. Device/inode identity is checked before and after receipt creation.
Exclusive descriptor-relative writes cannot follow a replacement pathname.
Short writes are completed; zero writes and storage errors fail closed. Receipt
files and their directory are fsynced. Failed receipt files are retained and must
not be blindly reused. Parent/ancestor directories must remain operator-trusted;
this is not a security boundary against root or a malicious process with the
same identity. Root-level bind-mount aliases are deliberately rejected.

**Container limitation:** the inactive Compose draft now uses the external
receipt paths, but its direct bind mount at that root is deliberately rejected
by the native mount-alias policy. CPU container commissioning therefore remains
blocked pending a separate reviewed mount-identity design. Do not weaken the
native check to make the draft run. No Docker command has been executed.

Stdout-only `emit_receipt` remains available for errors and non-conversion utility
reports; production converter/verifier CLIs do not allow omitting the receipt.
Neither receipt destination nor its ancestors may be automatically created.
The container-specific allowance is NOT implemented or implied.

## Lifecycle states

Machine-readable failures include phase, error_type and interruption information.
Conversion states include artifact/staging names, numerical_verification,
promoted and durability. The phases are:

- preflight: invalid source/options/receipt destination; no publication;
- conversion: streaming/storage failure, promoted=false;
- numerical_verification: independent verifier failure, promoted=false;
- pre_publication: verified result, staging sync/source recheck failure;
- atomic_promotion: syscall window, promoted=null if interrupted with uncertain outcome;
- post_promotion_fsync: promoted=true, durability=uncertain until parent fsync succeeds;
- published: promoted=true, durability=confirmed;
- receipt_write: carries the already published artifact's state even when receipt
  creation, sync or stdout reporting fails. No rollback or artifact deletion occurs.

A rename collision is a failed publication and never overwrites the existing
artifact. An interrupt precisely around rename cannot safely be guessed as
unpublished; inspect both names and independently verify. SIGKILL/power loss
cannot emit a Python receipt; recovery always requires inspection and verification.
File/directory fsync is the normal Linux durability contract, not protection from
faulty storage hardware. Partial or unsynced receipts must not be accepted solely
because their bytes resemble PASS; require successful command completion and
independent verification. Source/model file writes remain prohibited.

Use the README native commands, with external paths:
`--receipt /data/axonai/receipts/conversion-v2.json` and a different unused
`/data/axonai/receipts/verification-v2.json`. These commands remain unexecuted.

## Validation and performance

The complete updated suite covers the original exhaustive 65,536 BF16 patterns,
receipt protection and collisions, short/zero writes, ENOSPC, fsync errors,
interruptions before/after promotion, directory replacement, multidimensional and
scalar tensors, and simultaneous dtype/torch_dtype preservation. The original
manifest-anchored source requires dtype=bfloat16; conflicting torch_dtype remains
an error. Tests use synthetic fixtures under deployment/test-workspace only.
`test-results-v2-review.json` is the final test record. Earlier v2 reports include
an initial run with duplicate inherited cases; do not interpret that larger count
as additional coverage.

No independently installed safetensors reader is available. Official-reader
interoperability is UNAVAILABLE, not PASS; see interoperability-v2.json.
The integer numerical oracle is independent of casting but shares format I/O.

The 2 MiB synthetic benchmark measured approximately 0.895 MiB/s conversion and
0.874 MiB/s verification, each at about 100% of one CPU core. Process peak RSS
was 280172 KiB (~273.6 MiB), including interpreter/harness high-water accounting;
this is not an incremental allocation measurement or a full-model memory test.
Projection for 53,791,996,928 tensor bytes is 15.9 hours casting + 16.3 hours
internal verification (~32.2 hours), or ~48.5 hours including a fresh independent
verification. These are linear estimates from one in-memory run, excluding all
hashing and disk I/O, production data distribution and tenant contention; actual
elapsed time may be materially longer. No timing guarantee is made.

Python per-value struct packing/unpacking, integer arithmetic, dictionary counter
updates and float error calculations dominate. No numerical optimization was
made or authorized. Resource and independent runtime/ABI validation remain open.

No actual conversion, production weight payload reads, GPU/driver query, Docker
execution, installation, network access, AxonOS change or deployment occurred.
