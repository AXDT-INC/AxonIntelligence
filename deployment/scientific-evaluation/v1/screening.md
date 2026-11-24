# Private contamination screen v1

The screening custodian runs `screen.py` offline, under a separate account on an access-controlled volume outside Git. The script indexes the proposed pilot pool in memory and streams privately exported reference rows. Neither questions, keys, reference text, per-item reports, hashes of individual prompts, nor match exemplars belong in Git. Keep model operators and reviewers outside this volume. Files use mode 0600 under a directory with mode 0700. Supply authorized immutable copies. This phase used disposable synthetic fixtures only.

## Inventory and projections

Before items go to inference, two people with source and provenance access sign the inventory and family crosswalk in private custody. Reconcile every collection to approved source manifests. Record rights, artifact and export digests, projection code version, row counts, parentage, exclusions, missing sources, source-alias completeness and dates. Check the frozen corpus against 10,684 records and 4,230 scientific articles. Separately verify the 20,000-row staging artifact. Scientific articles and source families can have different counts. Candidate-pipeline rejects and auxiliary pools remain relevant even if they never reached SFT.

| Role | Minimum evidence to reconcile |
| --- | --- |
| `frozen_sft` | One canonical export in training-record units. Include identifying question, context and target-bearing text facets separately. |
| `staging` | One export of the original 20,000 staging rows. Preserve their distinct provenance even when some rows also occur in SFT. |
| `candidate_corpora` | Enumerate each available candidate pool and every material upstream branch, including inactive branches. Map rows to their original study/article. |
| `generated_variants` | Cover generated, rewritten, translated, filtered and discarded variants. Propagate parent source IDs and families. Include ancestor text as authorized. |
| `previous_evaluations` | Cover permitted prior prompts, sources and diagnostic variants, including unsuccessful evaluation attempts when input material existed. Catalog exposed cases even when no score exists. Use only authorized projections. |
| Historical sealed 51 | Keep the collection closed. Only previously authorized derived signals in separate custody can be used. The optional attestation interface below carries decisions by proposed pilot ID only. |

One JSONL row, in UTF-8, has exactly these four fields.

```json
{"record_id":"opaque-001","texts":["Synthetic blue bobbins turn beside six copper rings."],"source_ids":["urn:test:source1"],"source_family_ids":["urn:test:family1"]}
```

Use a stable opaque ASCII record ID within a collection. `texts` contains discriminating facets such as a complete question, the key-bearing rationale or a contextual passage. Exclude boilerplate and generic answer strings such as single multiple-choice letters. Split long documents into passages under one row. Include the same facets from the candidate construction workpapers. A source-free authored exercise has its own documented synthetic or author-origin source and family. Leave unknown ancestry unresolved. An actual unknown uses empty source lists, which put coverage on hold. Source IDs must be reconciled *before export*. Resolve PMCID/PMID/DOI aliases, preprint/publication versions, amendments, supplements, overlapping trial cohorts, reused datasets and derivative question templates. Each row carries all relevant canonical families. A version, translation, paraphrase or altered numeric parameter inherits the parent family.

Supply an inventory JSON object with the following layout. Bracketed values are placeholders for SHA-256 hex digests. Add each applicable collection. The minimum example intentionally leaves coverage incomplete.

```json
{
  "version": 1,
  "mode": "pilot",
  "inventory_audit_sha256": "<approved private inventory audit>",
  "family_crosswalk_sha256": "<audited private identity crosswalk>",
  "collections": [
    {"collection_id":"frozen","role":"frozen_sft","coverage":"complete","file":"frozen.jsonl","record_count":10684,"sha256":"<exact export bytes>","origin_sha256":"<approved source artifact or canonical manifest>"},
    {"collection_id":"unavailable-branch","role":"candidate_corpora","coverage":"missing"}
  ]
}
```

Mark partial exports `partial` and include their file, count and digest. Include a `missing` entry for each unavailable branch. In pilot mode the declared origin SHA-256 for frozen SFT must be `641b43e32f3d69fa6fbf74754772e976777a26383c27de2692b90670c5dc5d07` and the staging SHA-256 must be `24889f6495d4e9e5015f7a63352fba470b9913fcb1dbeebc7d84c0eb0546b5fd`. These pins come from the tracked frozen extraction receipt and staging verification result at the seven-commit baseline. Custodians hash the original bytes when constructing exports and bind their transformation to both hashes. This CLI validates the declared origins and hashes the exact projections it processes. It also checks unique record IDs and counts, and flags absent roles, missing or partial collections, empty exports and unverified crosswalks. An inventory can omit an unknown branch despite passing those checks. The inventory signers attest actual lineage and verify origin and audit pins against trusted records held separately. The scanner cannot authenticate those signers. `synthetic` mode waives the production origin pins and count expectations for unit tests only.

## Sealed history

Keep historical items, keys and mappings inaccessible to this workflow. Preserve the prior exposure restrictions on 049 and 015 and their derivatives. If a steward **already** has separately authorized derived denial data, the steward may compare only the proposed pilot export against those data within their own custody. A decision of `no_detected_overlap` requires the steward to attest adequate coverage of exact content, variants and family aliases across all 51. An exact-only denylist cannot clear an item for the other mechanisms. Do not open or regenerate the sealed material to fill gaps.

Keep `sealed_coverage` as a separate axis in every private registry entry and report. Use exactly one of the following routes for a particular pilot export.

| `sealed_coverage` | Evidence and disposition |
| --- | --- |
| `custodian_assessed` | The inventory’s `sealed_dispositions` descriptor contains `file` and `sha256`. Its document has exactly `scope` (the literal `historical_sealed_51`), `candidates_sha256` (exact candidate JSONL bytes), `attestation_sha256` (independent signed record), and `statuses` (a verdict for every candidate ID). No historical IDs or mappings enter this table. Each sealed verdict is retained as `sealed_status`. A positive or unresolved verdict blocks the affected candidate. |
| `unassessed` | No authorized table or approved pilot-scope decision. Sealed status remains null and an otherwise eligible negative is held as unresolved. |
| `pilot_scope_limited_unknown` | Where no prior sealed decisions exist for the export, an independent scientific lead and custody steward can approve the explicit diagnostic-only scope below. Sealed status remains null. The signed exception applies only to unknown overlap with the inaccessible cohort. All accessible-inventory checks and independent item reviews must still pass. |

For the limited scope, use inventory field `pilot_scope_disposition` with `file` and its SHA-256. The independent lead and steward inspect blinded authorship, sourcing and exposure records, establish that authors constructed the candidate pool independently, and document exclusion of **all** historical evaluation material and its derivatives from item design, including known exposures. They verify that no existing positive, unresolved or other sealed decisions are being suppressed. They separately verify signatures, roles and binding to the precise export in private custody. The scanner validates the decision document’s structure, pins, candidate digest, and coverage of all items. It relies on the external verification receipt for signatory authenticity.

```json
{
  "scope": "diagnostic_pilot_only",
  "candidates_sha256": "<exact pilot export>",
  "no_prior_sealed_decisions": true,
  "residual_risk": "historical_sealed_51_overlap_unknown",
  "signed_disposition_sha256": "<signed independent pilot-only approval>",
  "signature_verification_sha256": "<custody verification of signatures, roles and authorization>",
  "historical_material_exclusion_sha256": "<documented exclusion of historical material and derivatives>",
  "known_exposures_exclusion_sha256": "<verified exclusion of known exposed material>",
  "item_receipts": {
    "opaque-001": {
      "independent_authorship_sha256": "<independent authorship record>",
      "source_provenance_sha256": "<independently reviewed source lineage>"
    }
  }
}
```

Under this exception, `no_detected_overlap` covers the **inventoried accessible corpora only**. Carry `pilot_scope_limited_unknown` into registry records and pilot summaries, with the number of affected cases. Historical sealed overlap remains unknown. The exception grants no confirmation eligibility, no assurance of disjointness from sealed cases, and no permission to inspect the historical cohort. Supplying both decision routes, missing item receipts, a mismatched export, or an invalid decision digest aborts screening. A candidate with an actual positive or unresolved sealed verdict cannot use the exception, including after a pool is subdivided.

## Matching and disposition

Run this CLI only in the private environment after approval. All three filenames are simple basenames inside the private root.

```sh
python3 -B screen.py /approved/private/screening inventory.json pilot-candidates.jsonl screening-receipt.json
```

The report records script, inventory and candidate-export digests, coverage gaps, counts per collection role, and per-item signals and comparator references. It refuses to overwrite an earlier receipt. Counts refer to distinct comparator rows per candidate and status within a collection. The first 12 matches per status are retained as private examples, with total counts supplied independently. Restrict access to the report itself. A second report requires a new filename.

| Signal | Decision rule and action |
| --- | --- |
| `exact_match` | An entire text facet agrees after Unicode NFKC and whitespace normalization. Case and punctuation are preserved. Exclude confirmed reused material. An accidentally generic facet requires a documented projection correction and full rerun. |
| `shared_source` | At least one canonical source or family ID matches. A within-pool match flags both items. Select at most one member per source family using the predeclared construction rule and rescreen the final selection. Exclude intersections with covered training or historical families from the clean stratum. |
| `probable_paraphrase` | With at least six distinct alphanumeric tokens in each facet, token-set Jaccard ≥0.65, or at least three shared word trigrams and trigram containment ≥0.65 relative to the shorter trigram set. A reviewer checks the actual referents, logic and lineage. |
| `unresolved` | Intermediate Jaccard ≥0.48 or containment ≥0.45 with the same minimum features, unresolved sealed verdict, insufficient candidate or comparator lexical features, weak source identity, or an inventory gap without a stronger match. Request source reconciliation, classify as diagnostic, or replace. |
| `no_detected_overlap` | No configured positive or ambiguous signal across the authorized comparison scope and no missing accessible corpus. Interpret together with the sealed-coverage field. An independent screening disposition is still required. |

The report uses the table order for the immutable per-item `raw_status` and retains simultaneously observed signals. Global coverage gaps apply to every item. All candidates are also compared against one another, excluding self-matches. Token similarity can miss semantic restatements, translations and numerically respecified problems, and can flag ordinary disciplinary vocabulary. In private custody, calibrate retrieval on deliberately planted known duplicates, translated/semantic variants, short cases, sibling publications, numeric/code variations and unrelated controls. Record sensitivity by mechanism, false-positive rate, misses and review burden. Lock the version, thresholds and fallback before freezing the pilot. These unit tests verify program behavior and provide no estimate of scientific recall.

The independent screener must examine source-family and semantic coverage for every candidate, including lexical negatives. Use source search, domain review and authorized offline secondary retrieval if available. Preserve a signed item-level receipt recording the input and report digests, automatic signals, actual evidence examined, semantic/alias checks, disposition, rationale and reviewer role. The registry’s `contamination.status` is the final reviewed classification. Preserve the separate `sealed_coverage` value unchanged. Set `contamination.receipt` to the independent decision record and `contamination.disposition` to the corresponding custody decision. For example, an incidental vocabulary match can retain `probable_paraphrase` in its immutable raw report and receive reviewed status `no_detected_overlap` and disposition `approved_for_pilot` only after an independent reviewer documents distinct lineage, assesses the actual cases and verifies complete coverage. Changing a prompt, source list or corpus requires a new exported version and a new screen. Preserve both versions. True shared-family/exact cases, exposure, incomplete rights, missing corpora and unresolved semantic/alias coverage remain outside the clean comparison. Report the number screened, excluded, replaced and held, by reason. Apply all rules without model responses or model-arm identity. Unknown foundation-model pretraining data limits any claim to the documented inventories. A future holdout needs independent authors, new source families and a separately frozen screen that includes this pilot and its variants. No future holdout content enters this phase.
