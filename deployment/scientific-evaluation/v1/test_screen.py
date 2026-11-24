"""Disposable fixtures use arbitrary words and nonexistent source IDs."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import screen


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def record(key, text, source=None, family=None):
    return {"record_id": key, "texts": [text], "source_ids": [source or f"urn:test:{key}"],
            "source_family_ids": [family or f"test-family:{key}"]}


class PrivateScreenTests(unittest.TestCase):
    def write(self, name, raw):
        path = self.root / name
        path.write_bytes(raw)
        path.chmod(0o600)
        return digest(raw)

    def js(self, name, data):
        return self.write(name, json.dumps(data).encode())

    def rows(self, name, rows):
        return self.write(name, b"".join((json.dumps(row) + "\n").encode() for row in rows))

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.candidates = [record("p1", "crimson bobbins glide above the seven pewter cylinders")]
        self.references = {role: [record("r1", "umbra fig feather lamp hook wheel")] for role in screen.ROLES}

    def inventory(self):
        export = self.rows("pilot.jsonl", self.candidates)
        collections = []
        for role, rows in sorted(self.references.items()):
            name = role + ".jsonl"
            pin = self.rows(name, rows)
            collections.append({"collection_id": role, "role": role, "coverage": "complete",
                                "file": name, "sha256": pin, "origin_sha256": digest(role.encode()),
                                "record_count": len(rows)})
        sealed = {"scope": "historical_sealed_51", "candidates_sha256": export,
                  "attestation_sha256": digest(b"synthetic audit"),
                  "statuses": {r["record_id"]: "no_detected_overlap" for r in self.candidates}}
        pin = self.js("dispositions.json", sealed)
        return {"version": 1, "mode": "synthetic", "collections": collections,
                "inventory_audit_sha256": digest(b"synthetic inventory"),
                "family_crosswalk_sha256": digest(b"synthetic aliases"),
                "sealed_dispositions": {"file": "dispositions.json", "sha256": pin}}

    def run_screen(self, manifest):
        self.js("inventory.json", manifest)
        return screen.screen(self.root, "inventory.json", "pilot.jsonl")

    def scope_document(self):
        return {"scope": "diagnostic_pilot_only",
                "candidates_sha256": digest((self.root / "pilot.jsonl").read_bytes()),
                "no_prior_sealed_decisions": True, "residual_risk": "historical_sealed_51_overlap_unknown",
                "signed_disposition_sha256": digest(b"synthetic signed disposition"),
                "signature_verification_sha256": digest(b"synthetic detached verification"),
                "historical_material_exclusion_sha256": digest(b"synthetic exclusions"),
                "known_exposures_exclusion_sha256": digest(b"synthetic known exposures"),
                "item_receipts": {"p1": {"independent_authorship_sha256": digest(b"synthetic authorship"),
                                         "source_provenance_sha256": digest(b"synthetic sources")}}}

    def test_negative_is_review_required_and_no_raw_text_in_report(self):
        result = self.run_screen(self.inventory())
        self.assertEqual(result["missing_coverage"], [])
        self.assertEqual(result["items"][0]["raw_status"], "no_detected_overlap")
        self.assertEqual(result["items"][0]["sealed_coverage"], "custodian_assessed")
        self.assertEqual(result["items"][0]["sealed_status"], "no_detected_overlap")
        self.assertTrue(result["independent_review_required"])
        self.assertNotIn("crimson", json.dumps(result))
        screen.write_report(self.root, "receipt.json", result)
        path = self.root / "receipt.json"
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            screen.write_report(self.root, "receipt.json", {})

    def test_exact_shared_family_and_variant_referrals_stay_distinct(self):
        self.references["frozen_sft"] = [record("a", "crimson   bobbins glide above the seven pewter cylinders")]
        self.references["generated_variants"] = [
            record("b", "crimson bobbins glide above the seven copper cylinders")]
        self.references["previous_evaluations"] = [record("c", "plain spiral", family="test-family:p1")]
        result = self.run_screen(self.inventory())["items"][0]
        self.assertEqual(result["raw_status"], "exact_match")
        self.assertEqual(result["samples"]["shared_source"][0]["methods"], ["source_family"])
        collections = [m["collection"] for m in result["samples"]["probable_paraphrase"]]
        self.assertIn("generated_variants", collections)

    def test_pool_detects_both_sides_of_family_collision(self):
        self.candidates.append(record("p2", "twenty distant rings trace pale arcs", family="test-family:p1"))
        result = self.run_screen(self.inventory())["items"]
        self.assertEqual([r["raw_status"] for r in result], ["shared_source"] * 2)
        self.assertEqual(result[0]["samples"]["shared_source"][0]["collection"], "pilot_pool")

    def test_midrange_phrasing_requires_review(self):
        self.references["candidate_corpora"] = [
            record("r1", "crimson bobbins float below the seven pewter cylinders")]
        item = self.run_screen(self.inventory())["items"][0]
        self.assertEqual(item["raw_status"], "unresolved")

    def test_partial_coverage_and_missing_crosswalk_fail_closed(self):
        self.references["staging"][0]["source_ids"] = []
        manifest = self.inventory()
        del manifest["sealed_dispositions"], manifest["family_crosswalk_sha256"]
        manifest["collections"] = [c for c in manifest["collections"] if c["role"] != "previous_evaluations"]
        manifest["collections"][0]["coverage"] = "partial"
        report = self.run_screen(manifest)
        self.assertEqual(report["items"][0]["raw_status"], "unresolved")
        gaps = report["missing_coverage"]
        self.assertEqual(report["sealed_coverage"], "unassessed")
        for gap in ("previous_evaluations.absent",
                    "staging.unresolved_source_crosswalk", "unverified_family_crosswalk"):
            self.assertIn(gap, gaps)
        self.assertTrue(any(g.endswith(".partial") for g in gaps))

    def test_official_counts_apply_to_pilot_mode(self):
        manifest = self.inventory()
        manifest["mode"] = "pilot"
        with self.assertRaisesRegex(ValueError, "trusted corpus pin"):
            self.run_screen(manifest)
        for entry in manifest["collections"]:
            if entry["role"] in screen.ORIGIN_PINS:
                entry["origin_sha256"] = screen.ORIGIN_PINS[entry["role"]]
        report = self.run_screen(manifest)
        self.assertEqual(set(report["missing_coverage"]), {"frozen_sft.unexpected_total",
                                                           "staging.unexpected_total"})

    def test_changed_inputs_abort(self):
        manifest = self.inventory()
        self.rows("staging.jsonl", [record("other", "other input")])
        with self.assertRaisesRegex(ValueError, "digest or count"):
            self.run_screen(manifest)
        manifest = self.inventory()
        self.candidates[0]["texts"] = ["revised candidate"]
        self.rows("pilot.jsonl", self.candidates)
        with self.assertRaisesRegex(ValueError, "unbound"):
            self.run_screen(manifest)

    def test_sealed_table_permits_only_opaque_complete_dispositions(self):
        manifest = self.inventory()
        table = json.loads((self.root / "dispositions.json").read_bytes())
        table["statuses"] = {}
        manifest["sealed_dispositions"]["sha256"] = self.js("dispositions.json", table)
        with self.assertRaisesRegex(ValueError, "unbound"):
            self.run_screen(manifest)

    def test_scoped_pilot_decision_preserves_unknown_and_accessible_gates(self):
        manifest = self.inventory()
        del manifest["sealed_dispositions"]
        self.assertEqual(self.run_screen(manifest)["items"][0]["raw_status"], "unresolved")
        doc = self.scope_document()
        manifest["pilot_scope_disposition"] = {"file": "scope.json", "sha256": self.js("scope.json", doc)}
        item = self.run_screen(manifest)["items"][0]
        self.assertEqual(item["raw_status"], "no_detected_overlap")
        self.assertEqual(item["sealed_coverage"], "pilot_scope_limited_unknown")
        self.assertIsNone(item["sealed_status"])
        manifest["collections"][0]["coverage"] = "partial"
        self.assertEqual(self.run_screen(manifest)["items"][0]["raw_status"], "unresolved")
        doc["item_receipts"] = {}
        manifest["pilot_scope_disposition"]["sha256"] = self.js("scope.json", doc)
        with self.assertRaisesRegex(ValueError, "incomplete pilot scope coverage"):
            self.run_screen(manifest)

    def test_scoped_pilot_decision_cannot_replace_existing_denials(self):
        manifest = self.inventory()
        table = json.loads((self.root / "dispositions.json").read_bytes())
        for status in ("unresolved", "exact_match"):
            table["statuses"]["p1"] = status
            manifest["sealed_dispositions"]["sha256"] = self.js("dispositions.json", table)
            item = self.run_screen(manifest)["items"][0]
            self.assertEqual((item["raw_status"], item["sealed_status"]), (status, status))
        doc = self.scope_document()
        manifest["pilot_scope_disposition"] = {"file": "scope.json", "sha256": self.js("scope.json", doc)}
        with self.assertRaisesRegex(ValueError, "cannot override"):
            self.run_screen(manifest)
        del manifest["sealed_dispositions"]
        doc["candidates_sha256"] = digest(b"other candidate revision")
        manifest["pilot_scope_disposition"]["sha256"] = self.js("scope.json", doc)
        with self.assertRaisesRegex(ValueError, "unbound"):
            self.run_screen(manifest)
        del doc["signature_verification_sha256"]
        manifest["pilot_scope_disposition"]["sha256"] = self.js("scope.json", doc)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.run_screen(manifest)

    def test_unsafe_files_and_unidentified_sources(self):
        manifest = self.inventory()
        (self.root / "staging.jsonl").chmod(0o644)
        with self.assertRaisesRegex(ValueError, "private regular file"):
            self.run_screen(manifest)
        with self.assertRaises(ValueError):
            screen.named_file(self.root, "../other-file")
        self.candidates[0]["source_family_ids"] = []
        self.assertEqual(self.run_screen(self.inventory())["items"][0]["raw_status"], "unresolved")
        (self.root / ".git").mkdir()
        (self.root / ".git" / "HEAD").touch()
        with self.assertRaisesRegex(ValueError, "outside Git"):
            self.run_screen(manifest)


if __name__ == "__main__":
    unittest.main()
