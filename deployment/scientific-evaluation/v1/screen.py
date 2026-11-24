"""Custodian-only, streaming lexical and source screening. Standard library only."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import unicodedata


ROLES = {"frozen_sft", "staging", "candidate_corpora", "generated_variants", "previous_evaluations"}
ORIGIN_PINS = {
    "frozen_sft": "641b43e32f3d69fa6fbf74754772e976777a26383c27de2692b90670c5dc5d07",
    "staging": "24889f6495d4e9e5015f7a63352fba470b9913fcb1dbeebc7d84c0eb0546b5fd",
}
PRIORITY = ("exact_match", "shared_source", "probable_paraphrase", "unresolved")
NEGATIVE = "no_detected_overlap"
SEALED_SCOPE = "historical_sealed_51"
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
SAMPLE_LIMIT = 12
VERSION = "screen-v1.0"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def valid_digest(value):
    return isinstance(value, str) and DIGEST.fullmatch(value)


def distinct_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def parse(raw):
    return json.loads(raw.decode("utf-8"), object_pairs_hook=distinct_keys)


def private_root(path):
    root = Path(path).resolve(strict=True)
    require(root.is_dir() and root.stat().st_uid == os.getuid() and
            not root.stat().st_mode & 0o077, "expected owner-only private directory")
    require(all(not (p / ".git" / "HEAD").exists() and not (p / ".git").is_file()
                for p in (root, *root.parents)),
            "private directory must be outside Git")
    return root


def named_file(root, name, existing=True):
    require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name),
            "expected a simple filename within private directory")
    path = root / name
    if existing:
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and
                info.st_nlink == 1 and not info.st_mode & 0o077, "expected private regular file")
    return path


def string_list(values, allow_empty=True):
    return (isinstance(values, list) and (allow_empty or bool(values))
            and all(isinstance(v, str) and v.strip() for v in values))


def identity(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value)


def check_record(row):
    require(isinstance(row, dict) and set(row) == {"record_id", "texts", "source_ids", "source_family_ids"},
            "invalid record projection")
    require(identity(row["record_id"]) and string_list(row["texts"], False)
            and string_list(row["source_ids"]) and string_list(row["source_family_ids"]),
            "invalid private record fields")
    return row


def scan_rows(root, name, consume):
    digest = hashlib.sha256()
    ids = set()
    with named_file(root, name).open("rb") as stream:
        for raw in stream:
            row = check_record(parse(raw))
            require(row["record_id"] not in ids, "duplicate record ID within collection")
            ids.add(row["record_id"])
            digest.update(raw)
            consume(row)
    return digest.hexdigest(), len(ids)


def features(text):
    canonical = " ".join(unicodedata.normalize("NFKC", text).split())
    words = TOKEN.findall(canonical.casefold())
    triples = set(zip(words, words[1:], words[2:]))
    return canonical, set(words), triples


def compare_counts(shared_words, n_query, n_ref, shared_triples, n_query_triples, n_ref_triples):
    # Length scales vary across question, variant, and source-excerpt projections.
    jaccard = shared_words / max(1, n_query + n_ref - shared_words)
    containment = shared_triples / max(1, min(n_query_triples, n_ref_triples))
    if min(n_query, n_ref) >= 6:
        if jaccard >= 0.65 or (shared_triples >= 3 and containment >= 0.65):
            return "probable_paraphrase"
        if jaccard >= 0.48 or (shared_triples >= 3 and containment >= 0.45):
            return "unresolved"
    return None


class PilotIndex:
    """Index the small candidate pool; stream comparators one record at a time."""

    def __init__(self, rows):
        self.exact = defaultdict(set)
        self.sources = defaultdict(set)
        self.families = defaultdict(set)
        self.words = defaultdict(set)
        self.triples = defaultdict(set)
        self.lengths = {}
        self.results = {}
        for row in rows:
            item = row["record_id"]
            self.results[item] = {"item_id": item, "counts": Counter(), "samples": defaultdict(list),
                                  "quality_flags": []}
            if not row["source_ids"] or not row["source_family_ids"]:
                self.results[item]["quality_flags"].append("missing_candidate_source_or_family")
            rich = False
            for facet, text in enumerate(row["texts"]):
                canonical, words, triples = features(text)
                self.exact[canonical].add(item)
                key = (item, facet)
                self.lengths[key] = (len(words), len(triples))
                rich |= len(words) >= 6
                for word in words:
                    self.words[word].add(key)
                for triple in triples:
                    self.triples[triple].add(key)
            if not rich:
                self.results[item]["quality_flags"].append("insufficient_candidate_lexical_features")
            for source in row["source_ids"]:
                self.sources[source].add(item)
            for family in row["source_family_ids"]:
                self.families[family].add(item)

    def record_signal(self, item, status, collection, record, methods):
        result = self.results[item]
        result["counts"][status] += 1
        samples = result["samples"][status]
        if len(samples) < SAMPLE_LIMIT:
            samples.append({"collection": collection, "record_id": record, "methods": sorted(methods)})

    def compare(self, row, collection, same_pool=False):
        # Count once per (candidate, comparator, status), even when multiple fields match.
        signals = defaultdict(set)
        sufficiently_long = False
        for mapping, ids, method in ((self.sources, row["source_ids"], "source_id"),
                                     (self.families, row["source_family_ids"], "source_family")):
            for value in ids:
                for item in mapping.get(value, ()):
                    signals[item, "shared_source"].add(method)
        for text in row["texts"]:
            canonical, words, triples = features(text)
            sufficiently_long |= len(words) >= 6
            for item in self.exact.get(canonical, ()):
                signals[item, "exact_match"].add("normalized_full_text")
            words_shared = Counter(key for word in words for key in self.words.get(word, ()))
            triples_shared = Counter(key for part in triples for key in self.triples.get(part, ()))
            for key, overlap in words_shared.items():
                item, _ = key
                n_words, n_triples = self.lengths[key]
                status = compare_counts(overlap, n_words, len(words), triples_shared[key],
                                        n_triples, len(triples))
                if status:
                    signals[item, status].add("lexical_retrieval")
        for (item, status), methods in signals.items():
            if not same_pool or item != row["record_id"]:
                self.record_signal(item, status, collection, row["record_id"], methods)
        return sufficiently_long


def inspect_collection(root, entry, index, gaps, mode):
    role, label = entry["role"], entry["collection_id"]
    coverage = entry["coverage"]
    require(role in ROLES and identity(label) and label not in {"pilot_pool", SEALED_SCOPE}
            and coverage in ("complete", "partial", "missing"), "invalid collection description")
    if coverage != "complete":
        gaps.add(f"{label}.{coverage}")
    if coverage == "missing":
        return 0
    require(valid_digest(entry["origin_sha256"]) and valid_digest(entry["sha256"])
            and type(entry["record_count"]) is int and entry["record_count"] >= 0,
            "invalid collection provenance")
    if mode == "pilot" and role in ORIGIN_PINS:
        require(entry["origin_sha256"] == ORIGIN_PINS[role],
                "input origin declaration differs from trusted corpus pin")

    def consume(row):
        if not row["source_ids"] or not row["source_family_ids"]:
            gaps.add(f"{label}.unresolved_source_crosswalk")
        if not index.compare(row, label):
            gaps.add(f"{label}.insufficient_reference_lexical_features")

    digest, count = scan_rows(root, entry["file"], consume)
    require(digest == entry["sha256"] and count == entry["record_count"],
            "export digest or count differs from inventory")
    if not count:
        gaps.add(f"{label}.empty_export")
    return count


def read_decision(root, descriptor):
    require(isinstance(descriptor, dict) and set(descriptor) == {"file", "sha256"},
            "invalid decision document descriptor")
    raw = named_file(root, descriptor["file"]).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == descriptor["sha256"], "decision document digest mismatch")
    return parse(raw)


def check_scope_decision(decision, candidate_digest, item_ids):
    required_pins = {"signed_disposition_sha256", "signature_verification_sha256",
                     "historical_material_exclusion_sha256", "known_exposures_exclusion_sha256"}
    require(isinstance(decision, dict) and set(decision) == required_pins | {
        "scope", "candidates_sha256", "residual_risk", "no_prior_sealed_decisions", "item_receipts"
    } and decision["scope"] == "diagnostic_pilot_only"
            and decision["candidates_sha256"] == candidate_digest
            and decision["residual_risk"] == "historical_sealed_51_overlap_unknown"
            and decision["no_prior_sealed_decisions"] is True
            and all(valid_digest(decision[key]) for key in required_pins),
            "unbound or incomplete pilot scope decision")
    receipts = decision["item_receipts"]
    require(isinstance(receipts, dict) and set(receipts) == item_ids,
            "incomplete pilot scope coverage")
    for evidence in receipts.values():
        require(isinstance(evidence, dict) and
                set(evidence) == {"independent_authorship_sha256", "source_provenance_sha256"}
                and all(valid_digest(pin) for pin in evidence.values()),
                "incomplete pilot authorship or provenance evidence")


def sealed_coverage(root, manifest, candidate_digest, index):
    # Never consume historical item material. Each permitted route has distinct claim scope.
    descriptor = manifest.get("sealed_dispositions")
    scoped = manifest.get("pilot_scope_disposition")
    if scoped is not None:
        require(descriptor is None, "pilot scope cannot override custodian dispositions")
        check_scope_decision(read_decision(root, scoped), candidate_digest, set(index.results))
        return "pilot_scope_limited_unknown", {}
    if descriptor is None:
        return "unassessed", {}
    table = read_decision(root, descriptor)
    require(set(table) == {"scope", "candidates_sha256", "attestation_sha256", "statuses"}
            and table["scope"] == SEALED_SCOPE and table["candidates_sha256"] == candidate_digest
            and valid_digest(table["attestation_sha256"]) and isinstance(table["statuses"], dict)
            and set(table["statuses"]) == set(index.results), "unbound sealed disposition")
    for item, status in table["statuses"].items():
        require(status in (*PRIORITY, NEGATIVE), "unknown sealed disposition")
        if status != NEGATIVE:
            index.record_signal(item, status, SEALED_SCOPE, None, ["custodian_disposition"])
    return "custodian_assessed", table["statuses"]


def screen(root_path, inventory_file, candidates_file):
    root = private_root(root_path)
    inventory_raw = named_file(root, inventory_file).read_bytes()
    manifest = parse(inventory_raw)
    require(manifest["mode"] in ("pilot", "synthetic") and manifest["version"] == 1
            and type(manifest["version"]) is int and isinstance(manifest["collections"], list),
            "invalid inventory version, mode or collections")
    candidates = []
    candidate_digest, count = scan_rows(root, candidates_file, candidates.append)
    require(count > 0, "empty pilot candidate export")
    index = PilotIndex(candidates)
    for row in candidates:
        index.compare(row, "pilot_pool", same_pool=True)
    gaps = set()
    if not valid_digest(manifest.get("inventory_audit_sha256")):
        gaps.add("unattested_inventory")
    if not valid_digest(manifest.get("family_crosswalk_sha256")):
        gaps.add("unverified_family_crosswalk")
    seen, totals = set(), Counter()
    for entry in manifest["collections"]:
        require(entry["collection_id"] not in seen, "duplicate collection identifier")
        if entry["role"] in {"frozen_sft", "staging"}:
            require(entry["role"] not in totals, "canonical export role already registered")
        seen.add(entry["collection_id"])
        totals[entry["role"]] += inspect_collection(root, entry, index, gaps, manifest["mode"])
    for role in ROLES - set(totals):
        gaps.add(f"{role}.absent")
    if manifest["mode"] == "pilot":
        for role, expected in (("frozen_sft", 10684), ("staging", 20000)):
            if totals[role] != expected:
                gaps.add(f"{role}.unexpected_total")
    coverage, decisions = sealed_coverage(root, manifest, candidate_digest, index)
    for item in index.results.values():
        item["sealed_coverage"] = coverage
        item["sealed_status"] = decisions.get(item["item_id"])
        needs_resolution = gaps or item["quality_flags"] or coverage == "unassessed"
        item["raw_status"] = next((s for s in PRIORITY if item["counts"][s]),
                                  "unresolved" if needs_resolution else NEGATIVE)
    return {"version": VERSION, "mode": manifest["mode"],
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "inventory_sha256":
            hashlib.sha256(inventory_raw).hexdigest(), "candidates_sha256": candidate_digest,
            "sealed_coverage": coverage,
            "records_by_role": dict(totals), "missing_coverage": sorted(gaps),
            "independent_review_required": True, "items": list(index.results.values())}


def write_report(root, name, result):
    dest = named_file(root, name, existing=False)
    with tempfile.NamedTemporaryFile(dir=root, mode="w", encoding="utf-8") as tmp:
        json.dump(result, tmp, allow_nan=False, sort_keys=True)
        tmp.write("\n")
        tmp.flush()
        os.fsync(tmp.fileno())
        os.link(tmp.name, dest)  # Fails atomically if a report already exists.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("private_root", "inventory", "candidates", "report"):
        parser.add_argument(arg)
    args = parser.parse_args()
    try:
        root = private_root(args.private_root)
        result = screen(root, args.inventory, args.candidates)
        write_report(root, args.report, result)
    except (KeyError, TypeError, ValueError, OSError):
        # Never emit raw texts, private paths, or arbitrary JSON values into scheduler logs.
        parser.exit(1, "Screening failed; verify the private inputs, permissions and receipt pins.\n")
    print("Private screening report written.")


if __name__ == "__main__":
    main()
