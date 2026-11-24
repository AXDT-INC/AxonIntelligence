"""CPU-only synthetic score and paired analysis fixtures (in-memory only)."""
import hashlib
import json
import unittest

from analysis import agreement, analyze, plan_pairs
from score import final_score, score


class ScoreTest(unittest.TestCase):
    def test_numeric_unit_specific_inclusive_tolerance(self):
        key = {"kind": "numeric", "rtol": ".01",
               "targets": {"U": {"value": "10", "atol": ".05"}, "mU": {"value": "10000", "atol": "50"}}}
        for value, unit, errors in [("10.1", "U", []), ("10.10000001", "U", ["numeric_error"]),
                                    ("10100", "mU", []), ("10", "x", ["wrong_unit"]),
                                    ("NaN", "U", ["format_error"]), ("1e9999999", "U", ["format_error"])]:
            result = score(key, json.dumps({"value": value, "unit": unit}))
            self.assertEqual(result["errors"], errors)
            self.assertEqual(result["pass"], int(not errors))

    def test_choice_format_strict(self):
        key = {"kind": "multiple_choice", "options": ["a", "b"], "answer_id": "a"}
        self.assertEqual(score(key, '{"answer_id":"b","answer_id":"a"}')["errors"], ["format_error"])
        self.assertEqual(score(key, 'The answer is a')["pass"], 0)
        self.assertEqual(score(key, '{"answer_id":"b"}')["errors"], ["wrong_answer"])
        self.assertEqual(score(key, '{"answer_id":"a"}')["pass"], 1)
        self.assertEqual(score(key, '{"answer_id":[] }')["errors"], ["format_error"])
        with self.assertRaises(ValueError):
            score(dict(key, answer_id="missing"), "bad output")

    def test_claim_attribution_and_reference_compliance(self):
        key = {"kind": "evidence", "answer_id": "Y", "source_ids": ["x", "y", "z"],
               "support": {"c": [["x"], ["y", "z"]]}, "required": ["c"]}
        base = {"answer_id": "Y", "claims": [{"id": "c", "sources": ["z", "y"]}]}
        self.assertEqual(score(key, json.dumps(base))["pass"], 1)
        base["claims"][0]["sources"] = ["x", "z"]
        self.assertEqual(score(key, json.dumps(base))["errors"], ["attribution_error"])
        base["claims"] = [{"id": "d", "sources": ["made-up"]}]
        self.assertEqual(score(key, json.dumps(base))["errors"],
                         ["missing_required_claim", "unknown_reference", "unsupported_claim"])
        key["kind"] = "citation"
        del key["answer_id"], base["answer_id"]
        base["claims"] = [{"id": "c", "sources": []}]
        self.assertEqual(score(key, json.dumps(base))["errors"], ["attribution_error", "missing_citation"])

    def test_abstention_balanced(self):
        unknown = {"kind": "abstention", "answer_ids": [], "reason_ids": ["M"]}
        known = dict(unknown, answer_ids=["D"], reason_ids=[])
        yes = '{"decision":"answer","answer_id":"D"}'
        no = '{"decision":"abstain","reason_id":"M"}'
        self.assertEqual([score(k, a)["pass"] for k, a in
                          [(unknown, no), (unknown, yes), (known, no), (known, yes)]], [1, 0, 0, 1])

    def test_combined_review_requires_all_components(self):
        key = {"kind": "multiple_choice", "options": ["x", "y"], "answer_id": "x"}
        obj = score(key, '{"answer_id":"x"}')
        self.assertIsNone(final_score(obj, dual_review=True, final_rating=None)["pass"])
        self.assertEqual(final_score(obj, dual_review=True, final_rating=1)["pass"], 0)
        self.assertEqual(final_score(obj, dual_review=True, final_rating=2)["pass"], 1)
        self.assertEqual(final_score(score(key, " "), dual_review=True, final_rating=3)["errors"], ["empty_answer"])
        for response in (" ", '{"answer_id":"y"}'):
            failed = score(key, response)
            self.assertEqual(final_score(failed, dual_review=True, final_rating=None), failed)
        self.assertEqual(final_score(score({"kind": "expert"}, "toy"), dual_review=True, final_rating=2)["pass"], 1)
        with self.assertRaises(ValueError):
            final_score(obj, dual_review=False, final_rating=2)
        with self.assertRaises(ValueError):
            final_score({"status": "scored", "pass": 0, "errors": []}, dual_review=True, final_rating=3)

    def test_sandbox_binding_and_unavailable(self):
        # Opaque synthetic bytes. Nothing is executed.
        key = {"kind": "programming", "suite_sha256": "1"*64, "profile_sha256": "2"*64, "tests_total": 2}
        raw = '{"tests_passed":2}'
        self.assertEqual(score(key, raw)["status"], "sandbox_unavailable")
        receipt = {"suite_sha256": key["suite_sha256"], "profile_sha256": key["profile_sha256"],
                   "response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                   "outcome": "completed", "tests_passed": 1}
        self.assertEqual(score(key, raw, verified_sandbox_receipt=receipt)["errors"], ["test_failure"])
        receipt["tests_passed"] = 2
        self.assertEqual(score(key, raw, verified_sandbox_receipt=receipt)["pass"], 1)
        receipt["outcome"] = "infra_error"
        self.assertIsNone(score(key, raw, verified_sandbox_receipt=receipt)["pass"])
        with self.assertRaises(ValueError):
            score(key, raw + ' ', verified_sandbox_receipt=receipt)
        self.assertEqual(score({"kind": "expert"}, "free response")["status"], "pending_review")


def roster(*sources):
    return [{"item_id": str(i), "family": "toy", "source_family": s} for i, s in enumerate(sources)]


def row(i, arm, value, **kw):
    return {"item_id": str(i), "arm": arm, "pass": value, "status": "scored",
            "errors": [] if value else ["wrong_answer"], **kw}


class AnalysisTest(unittest.TestCase):
    def test_abstention_opportunities_keep_invalid_and_missing(self):
        r = roster("a", "b", "c", "d")
        for i, item in enumerate(r):
            item.update(family="uncertainty_abstention", answerability=(
                "answerable" if i < 2 else "insufficient_evidence"))
        data = [row(0, "A", 0, errors=["unnecessary_abstention"]), row(1, "A", 0, errors=["format_error"]),
                row(2, "B", 0, errors=["failure_to_abstain"]), row(3, "B", 0, errors=["wrong_abstention_reason"])]
        report = analyze(r, data)["abstention_opportunities"]
        over = report["answerable"]["arms"]["A"]
        self.assertEqual((over["events"], over["interpretable"], over["invalid_or_empty"]), (1, 1, 1))
        self.assertEqual(over["all_assigned_rate_bounds"], [.5, 1])
        under = report["insufficient_evidence"]["arms"]["B"]
        self.assertEqual(under["rate_among_interpretable"], .5)
        absent = report["answerable"]["arms"]["B"]
        self.assertEqual((absent["unscored"], absent["rate_among_interpretable"]), (2, None))
        self.assertEqual(absent["all_assigned_rate_bounds"], [0, 1])
        del r[0]["answerability"]
        with self.assertRaises(ValueError):
            analyze(r, data)

    def test_roster_missingness_and_cost(self):
        r = roster("a", "b", "c", "d", "e")
        data = [row(0, "A", 0, new_tokens=8, latency_s=2), row(0, "B", 1, new_tokens=2, latency_s=1),
                row(1, "A", 1), row(1, "B", 0), row(2, "A", 1), row(2, "B", 1), row(3, "A", 0),
                {"item_id": "3", "arm": "B", "pass": None, "status": "inference_failure",
                 "errors": ["oom"]}]
        report = analyze(r, data)
        overall = report["overall"]
        self.assertEqual((overall["assigned"], overall["complete"], overall["incomplete"]), (5, 3, 2))
        self.assertEqual((overall["B_wins"], overall["B_losses"], overall["ties_both_pass"]), (1, 1, 1))
        self.assertEqual(overall["B_minus_A"], 0)
        self.assertEqual(overall["all_assigned_missing_score_bounds"], [-.2, .4])
        self.assertEqual(report["arms"]["B"]["inference_failure_rate_assigned"], .2)
        self.assertEqual(report["arms"]["B"]["status_counts"]["missing_result"], 1)
        self.assertEqual(report["efficiency"]["tokens_per_generation_second"], {"paired": 1, "A": 4, "B": 2})
        self.assertEqual(report["efficiency"]["new_tokens"]["paired_B_minus_A"]["mean"], -6)

    def test_missing_and_identical_answers(self):
        self.assertIsNone(analyze(roster("a"), [])["overall"]["B_minus_A"])
        self.assertEqual(analyze(roster("a"), [])["overall"]["all_assigned_missing_score_bounds"], [-1, 1])
        r = roster(*map(str, range(80)))
        results = [row(i, arm, 1) for i in range(80) for arm in ("A", "B")]
        independent = analyze(r, results)["overall"]
        cluster = analyze(roster(*(["one"] * 80)), results)["overall"]
        self.assertLess(independent["complete_cluster_ci95"][1], cluster["complete_cluster_ci95"][1])
        self.assertGreater(independent["complete_cluster_ci95"][1], 0)
        self.assertGreater(independent["discordance_upper95_for_planning"], 0)
        self.assertEqual(independent["ties_both_pass"], 80)

    def test_duplicate_and_invalid_results(self):
        a = row(0, "A", 1)
        for data in ([a, a], [dict(a, item_id="9")], [dict(a, new_tokens=-1)], [dict(a, latency_s=0)],
                     [dict(a, status="pending_review")], [dict(a, arm="X")], [dict(a, **{"pass": True})]):
            with self.assertRaises(ValueError):
                analyze(roster("a"), data)
        with self.assertRaises(ValueError):
            analyze(roster("a") * 2, [])

    def test_agreement_with_missing_and_degenerate_labels(self):
        report = agreement([(0, 0), (1, 1), (0, 1), (1, 0), (None, 1)], levels=2)
        self.assertEqual((report["assigned"], report["double_rated"]), (5, 4))
        self.assertEqual((report["exact_agreement"], report["kappa"], report["linear_weighted_kappa"]), (.5, 0, 0))
        self.assertIsNone(agreement([(2, 2)] * 3)["kappa"])
        self.assertIsNone(agreement([(None, 1)])["exact_agreement"])
        self.assertEqual(agreement([(0, 0), (3, 3), (1, 1)])["linear_weighted_kappa"], 1)
        with self.assertRaises(ValueError):
            agreement([(4, None)])

    def test_power_sensitivity(self):
        small = plan_pairs(.2, .05, attrition=.1)
        large = plan_pairs(.4, .05)
        self.assertGreater(large["complete_independent_pairs"], small["complete_independent_pairs"])
        self.assertGreater(small["enroll"], small["complete_independent_pairs"])
        with self.assertRaises(ValueError):
            plan_pairs(.04, .05)


if __name__ == "__main__":
    unittest.main()
