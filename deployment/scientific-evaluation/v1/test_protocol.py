"""Fabricated registry and packets only. All test records are disposable."""
import hashlib
import json
import tarfile
import tempfile
import unittest
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from jsonschema import Draft7Validator

import analysis
import blind_review as review
import handoff
import pilot
import runner
import score
import verify_stage0 as stage0
from test_runner import SyntheticTokenizer


def entry(family, n):
    ident = f"{n:032x}"
    kind = {"scientific_programming": "programming", "uncertainty_abstention": "abstention",
            "evidence_qa": "evidence", "hallucination_citations": "citation",
            "experimental_design": "expert"}.get(family, "multiple_choice")
    row = {
        "protocol": "axonai-sci-v1", "item_id": ident, "revision": 1, "task_family": family,
        "cohort": "pilot_candidate", "source_ids": ["synthetic-source-" + ident],
        "source_family_id": "synthetic-family-" + ident,
        "rights": {"status": "cleared_internal_evaluation", "basis": "synthetic", "use_receipt": "r"},
        "contamination": {"status": "no_detected_overlap", "disposition": "approved_for_pilot", "receipt": "c"},
        "sealed_coverage": "pilot_scope_limited_unknown", "author_review": "independently_verified",
        "prior_exposure": "none_known", "expected_format": "fixture format",
        "scoring": {"kind": kind, "rule_ref": "k", "failure_taxonomy": "axonai-sci-v1", "dual_review": True},
        "content_commitment": "0" * 64,
    }
    if family == "uncertainty_abstention":
        row["answerability"] = "answerable" if n % 2 else "insufficient_evidence"
    return row


def frame():
    rows = []
    for family, quota in pilot.QUOTAS.items():
        for _ in range(quota + 2):
            rows.append(entry(family, len(rows)))
    return rows


def lock(rows):
    return hashlib.sha256(pilot.canonical(sorted(rows, key=lambda r: r["item_id"]))).hexdigest()


def draw(rows):
    return pilot.draw(rows, locked_frame_sha256=lock(rows), seed=b"x" * 32)


class PilotProtocol(unittest.TestCase):
    def test_real_design_counts_and_seeded_draw(self):
        rows = frame()
        a, b = draw(rows), draw(list(reversed(rows)))
        self.assertEqual(a, b)
        self.assertEqual({family: len(ids) for family, ids in a["selected"].items()}, pilot.QUOTAS)
        self.assertEqual(sum(map(len, a["selected"].values())), 80)
        group = set(a["selected"]["uncertainty_abstention"])
        self.assertEqual(sum(r.get("answerability") == "answerable" for r in rows if r["item_id"] in group), 2)
        selected = {i for ids in a["selected"].values() for i in ids}
        reserves = {i for ids in a["reserves"].values() for i in ids}
        self.assertTrue(selected.isdisjoint(reserves))
        self.assertEqual(len(selected | reserves), 100)
        first, second = pilot.paired_blocks(a["selected"], b"y" * 32)
        self.assertEqual((len(first), len(second)), (40, 40))
        self.assertEqual(set(first + second), selected)
        for ids in a["selected"].values():
            self.assertLessEqual(abs(len(set(first) & set(ids)) - len(set(second) & set(ids))), 1)

    def test_draw_rejects_drift_siblings_and_misclassified_items(self):
        rows = frame()
        with self.assertRaisesRegex(ValueError, "frame lock"):
            pilot.draw(rows, locked_frame_sha256="f" * 64, seed=b"x" * 32)
        rows[1]["source_family_id"] = rows[0]["source_family_id"]
        with self.assertRaisesRegex(ValueError, "clusters"):
            draw(rows)
        for family in ("scientific_programming", "uncertainty_abstention"):
            row = entry(family, 1)
            row["scoring"]["kind"] = "multiple_choice"
            self.assertFalse(pilot.VALIDATE.is_valid(row))

    def test_exposure_rights_and_unknown_coverage_exclude(self):
        item = entry("biomedical_factual", 0)
        for key, val in (("prior_exposure", "exposed"), ("sealed_coverage", "unassessed"),
                         ("cohort", "development_diagnostic"), ("author_review", "pending")):
            self.assertFalse(pilot.eligible({**item, key: val}))
        for key, field, val in (("rights", "status", "denied"), ("contamination", "status", "unresolved"),
                                ("contamination", "disposition", "pending")):
            row = deepcopy(item)
            row[key][field] = val
            self.assertFalse(pilot.eligible(row))
        rows = frame()
        for row in rows:
            if row["task_family"] == "uncertainty_abstention" and row["answerability"] == "answerable":
                row["cohort"] = "development_diagnostic"
        with self.assertRaisesRegex(ValueError, "insufficient eligible items"):
            draw(rows)

    def test_verifies_bundle_digest_and_answerability(self):
        row = entry("uncertainty_abstention", 1)
        bundle = {"item_id": row["item_id"], "revision": 1, "rule_ref": "k", "question": "fixture",
                  "visible_rubric": "fixture", "evidence_ref": "synthetic-snapshot",
                  "rights_ref": "r", "screen_ref": "c", "author_ref": "t",
                  "rule": {"kind": "abstention", "reason_ids": [], "answer_ids": ["v"]}}
        key = b"synthetic-only-commitment".ljust(32)
        row["content_commitment"] = pilot.bundle_commitment(row, bundle, key)
        pilot.verify_bundle(row, bundle, key)
        with self.assertRaisesRegex(ValueError, "differs from freeze"):
            pilot.verify_bundle(row, {**bundle, "question": "edited fixture"}, key)
        row["answerability"] = "insufficient_evidence"
        with self.assertRaisesRegex(ValueError, "differs from freeze"):
            pilot.verify_bundle(row, bundle, key)
        row["content_commitment"] = pilot.bundle_commitment(row, bundle, key)
        with self.assertRaisesRegex(ValueError, "stratum"):
            pilot.verify_bundle(row, bundle, key)


class ReviewProtocol(unittest.TestCase):
    def test_packets_cannot_be_swapped_or_adjudicated_by_raters(self):
        case = review.Case("opaque-response", "fixture", "", "fixture", "fixture")
        deliveries, ledger = review.prepare([case], ["r1", "r2"])
        p1, p2 = deliveries["r1"][0], deliveries["r2"][0]
        self.assertNotEqual(p1.ticket, p2.ticket)
        self.assertEqual(set(asdict(p1)), {"ticket", "question", "source_packet", "visible_rubric", "response"})
        r1, r2 = review.Rating(p1.ticket, 1, "missing", False), review.Rating(p2.ticket, 2, "ok", False)
        with self.assertRaises(ValueError):
            review.lock_ratings(ledger, [("r2", r1), ("r1", r2)])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            review.lock_ratings(ledger, [("r1", r1)])
        locked = review.lock_ratings(ledger, [("r1", r1), ("r2", r2)])
        ruling = review.Rating(case.response_id, 2, "criterion met", False)
        with self.assertRaisesRegex(ValueError, "adjudicator conflict"):
            review.finalize(ledger, locked, {case.response_id: ("r1", ruling)})
        self.assertEqual(review.finalize(ledger, locked, {case.response_id: ("r3", ruling)}),
                         {case.response_id: 2})
        self.assertEqual(locked[p1.ticket], r1)

    def test_scored_pair_from_rendering_through_final_join(self):
        registry = entry("experimental_design", 1)
        run_id, slots = "e" * 32, ["d" * 32, "c" * 32]
        runtime = runner.Runtime(*["0" * 64] * 5, (0,), (
            "GPU-00000000-0000-0000-0000-000000000001", "GPU-00000000-0000-0000-0000-000000000002"))
        item = runner.render_item(registry["item_id"], "synthetic token fixture", SyntheticTokenizer())
        receipts = []
        for slot, word in zip(slots, ("left", "right")):
            def decode(ids, limit):
                return runner.Decode(runner.digest(ids), (4, 0), word,
                                     runner.Measurement(.5, .1, (10, 10), (11, 11), 100))
            receipts.append(runner.run_item(run_id, slot, item, runtime, runtime, decode))
        self.assertEqual(runner.audit_pairs([item], receipts, run_id, slots, runtime), 2)
        aliases = ["anon-one", "anon-two"]
        cases = [review.Case(alias, "synthetic fixture", "", "fixture only", rec["text"])
                 for alias, rec in zip(aliases, receipts)]
        packets, ledger = review.prepare(cases, ["first", "second"])
        ratings = [(who, review.Rating(p.ticket, 2 if p.response == "right" else 1, "fixture", False))
                   for who, delivery in packets.items() for p in delivery]
        final = review.finalize(ledger, review.lock_ratings(ledger, ratings), {})
        schema = json.loads(Path(__file__).with_name("outcome.schema.json").read_text())
        Draft7Validator.check_schema(schema)
        validator = Draft7Validator(schema)
        results, envelopes = [], []
        for rec, alias, arm in zip(receipts, aliases, ("A", "B")):
            verdict = score.final_score(score.score({"kind": "expert"}, rec["text"]),
                                        dual_review=True, final_rating=final[alias])
            envelope = {"binding": handoff.score_binding(rec), "verdict": verdict}
            envelopes.append(envelope)
            results.append(handoff.analysis_row(rec, envelope, arm))
            validator.validate(results[-1])
        with self.assertRaisesRegex(ValueError, "bind"):
            handoff.analysis_row(receipts[0], envelopes[1], "A")
        result = analysis.analyze(handoff.analysis_roster([registry]), results)
        self.assertEqual(result["overall"]["B_wins"], 1)
        self.assertEqual(result["efficiency"]["new_tokens"]["paired"], 1)

    def test_technical_failure_remains_missing_and_blank_counts_as_zero(self):
        rec = {"item_id": "2"*32, "run_id": "1"*32, "slot_ticket": "3"*32, "output_sha256": None,
               "status": "worker_lost"}
        self.assertIsNone(handoff.analysis_row(rec, None, "A")["pass"])
        rec.update(status="empty_output", output_sha256="abc", output_tokens=1,
                   measurement={"latency_s": .5}, stop_reason="eos")
        envelope = {"binding": handoff.score_binding(rec), "verdict": score.score({"kind": "expert"}, "")}
        self.assertEqual(handoff.analysis_row(rec, envelope, "A")["pass"], 0)
        envelope["verdict"] = {"status": "pending_review", "pass": None, "errors": []}
        with self.assertRaisesRegex(ValueError, "blank"):
            handoff.analysis_row(rec, envelope, "A")


class EvidenceVerification(unittest.TestCase):
    def test_both_package_representations_checked_against_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent / stage0.PACKAGE
            root.mkdir()
            manifest = []
            for name in sorted(stage0.FILES):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                data = {}
                if name.endswith("DESCRIPTIVE_SUMMARY.v1.json"):
                    data = {"pair_preference_counts": {"base_preferred": 2, "finetuned_preferred": 1,
                            "no_clear_difference": 3}, "review_type": "fixture", "independent_expert_adjudication": False}
                elif name.endswith("COMPARISON_RESULTS.v1.json"):
                    data = {"outputs_generated": 0, "scores_assigned": 0}
                path.write_bytes(pilot.canonical(data))
                manifest.append(f"{stage0.digest(path.read_bytes())}  {name}\n")
            sums = root / "SHA256SUMS"
            sums.write_text("".join(manifest))
            archive = parent / (root.name + ".tar.gz")
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(root, arcname=root.name)
            with patch.object(stage0, "MANIFEST", stage0.digest(sums.read_bytes())), patch.object(
                    stage0, "ARCHIVE", stage0.digest(archive.read_bytes())):
                self.assertEqual(stage0.verify(parent)["paired_items"], 6)
                (root / "REPORT.md").write_text("changed synthetic material")
                with self.assertRaisesRegex(ValueError, "digest mismatch"):
                    stage0.verify(parent)
