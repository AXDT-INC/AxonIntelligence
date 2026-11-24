"""Custodian-only, deterministic pilot draw from a locked private sampling frame.

Library only. Callers retain the inputs, return values and HMAC key outside Git.
Requires the existing jsonschema Draft 7 implementation for item validation.
"""
import hashlib
import hmac
import json
from collections import defaultdict
from pathlib import Path

from jsonschema import Draft7Validator
from score import validate_key

SCHEMA = json.loads(Path(__file__).with_name("item.schema.json").read_text())
VALIDATE = Draft7Validator(SCHEMA)
QUOTAS = {
    "biomedical_factual": 12, "biomedical_reasoning": 12, "evidence_qa": 12,
    "hallucination_citations": 12, "experimental_design": 6, "statistical_reasoning": 6,
    "scientific_programming": 6, "uncertainty_abstention": 4, "general_retention": 5,
    "scientific_transfer": 5,
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def eligible(row):
    return (row["cohort"] == "pilot_candidate" and row["prior_exposure"] == "none_known"
            and row["rights"]["status"] == "cleared_internal_evaluation"
            and row["contamination"]["status"] == "no_detected_overlap"
            and row["contamination"]["disposition"] == "approved_for_pilot"
            and row["sealed_coverage"] != "unassessed"
            and row["author_review"] == "independently_verified")


def bundle_commitment(row, bundle, commitment_key):
    """Bind the private bundle and all registry decisions to the same immutable revision."""
    if type(commitment_key) is not bytes or len(commitment_key) < 32:
        raise ValueError("private commitment key required")
    metadata = {key: value for key, value in row.items() if key != "content_commitment"}
    data = canonical({"registry": metadata, "bundle": bundle})
    return hmac.digest(commitment_key, data, "sha256").hex()


def verify_bundle(row, bundle, commitment_key):
    """Custody-only preflight on separately authorized bytes. Reconcile referenced receipts too."""
    if not VALIDATE.is_valid(row):
        raise ValueError("invalid registry entry")
    expected = bundle_commitment(row, bundle, commitment_key)
    if not hmac.compare_digest(row["content_commitment"], expected):
        raise ValueError("item bundle differs from freeze")
    if (bundle["item_id"], bundle["revision"]) != (row["item_id"], row["revision"]):
        raise ValueError("wrong item revision")
    for field in ("question", "visible_rubric", "evidence_ref", "rights_ref", "screen_ref", "author_ref"):
        if not isinstance(bundle.get(field), str) or not bundle[field].strip():
            raise ValueError("missing item or clearance material")
    if (row["rights"]["use_receipt"], row["contamination"]["receipt"], row["scoring"]["rule_ref"]) != (
        bundle["rights_ref"], bundle["screen_ref"], bundle["rule_ref"]
    ):
        raise ValueError("clearance or rule binding differs")
    key = bundle["rule"]
    validate_key(key)
    if key["kind"] != row["scoring"]["kind"]:
        raise ValueError("registry and private scorer kind differ")
    if row["task_family"] == "uncertainty_abstention":
        is_answerable = bool(key["answer_ids"])
        if is_answerable != (row["answerability"] == "answerable"):
            raise ValueError("stratum and private abstention key differ")


def draw(frame, *, locked_frame_sha256, seed):
    """Return selected item IDs and ordered reserves. Generate the seed after frame lock."""
    if type(seed) is not bytes or len(seed) != 32:
        raise ValueError("32-byte independent draw seed required")
    frame = list(frame)
    groups = defaultdict(list)
    seen_ids, seen_sources, seen_families = set(), set(), set()
    for row in frame:
        if not VALIDATE.is_valid(row):
            raise ValueError("registry schema mismatch")
        ident, family = row["item_id"], row["task_family"]
        if ident in seen_ids:
            raise ValueError("multiple entries for one identity")
        seen_ids.add(ident)
        if not eligible(row):
            continue
        if ((row["scoring"]["kind"] in {"expert", "evidence", "citation"}
             or family in {"biomedical_reasoning", "experimental_design", "evidence_qa", "hallucination_citations"})
            and not row["scoring"]["dual_review"]):
            raise ValueError("expert review required")
        if seen_sources.intersection(row["source_ids"]) or row["source_family_id"] in seen_families:
            raise ValueError("resolve source clusters before locking the frame")
        seen_sources.update(row["source_ids"])
        seen_families.add(row["source_family_id"])
        stratum = row["answerability"] if family == "uncertainty_abstention" else "all"
        groups[family, stratum].append(ident)

    commitment = hashlib.sha256(canonical(sorted(frame, key=lambda row: row["item_id"]))).hexdigest()
    if not hmac.compare_digest(commitment, locked_frame_sha256):
        raise ValueError("frame lock mismatch")
    selected, reserves = {}, {}
    for family, quota in QUOTAS.items():
        quotas = {"all": quota}
        if family == "uncertainty_abstention":
            quotas = {"answerable": 2, "insufficient_evidence": 2}
        selected[family], reserves[family] = [], []
        for stratum, n in quotas.items():
            if len(groups[family, stratum]) < n:
                raise ValueError(f"insufficient eligible items in {family}/{stratum}")
            ordered = sorted(groups[family, stratum], key=lambda ident: (
                hmac.digest(seed, b"pilot-v1:" + ident.encode("ascii"), "sha256"), ident))
            selected[family].extend(ordered[:n])
            reserves[family].extend(ordered[n:])
    return {"frame_sha256": commitment, "quotas": QUOTAS.copy(), "selected": selected,
            "reserves": reserves}


def paired_blocks(selected, order_seed):
    """Custodian-only two balanced blocks. Independently randomize ticket-to-artifact assignment."""
    if type(order_seed) is not bytes or len(order_seed) != 32 or set(selected) != set(QUOTAS):
        raise ValueError("invalid block seed or family roster")
    def rank(label):
        return hmac.digest(order_seed, label.encode("ascii"), "sha256")

    odd = sorted((family for family, n in QUOTAS.items() if n % 2), key=rank)
    first_extra = set(odd[::2])
    blocks = ([], [])
    for family, quota in QUOTAS.items():
        ids = selected[family]
        if len(ids) != quota:
            raise ValueError("block assignment must use the locked family quotas")
        ordered = sorted(ids, key=lambda item: rank("split/" + item))
        cut = quota // 2 + int(family in first_extra)
        blocks[0].extend(ordered[:cut])
        blocks[1].extend(ordered[cut:])
    if len(set(blocks[0] + blocks[1])) != sum(QUOTAS.values()):
        raise ValueError("duplicate item in block assignment")
    return [sorted(block, key=lambda item: rank("order/" + item)) for block in blocks]
