"""Private, arm-blind scoring. No model, filesystem, network or code execution."""
import hashlib
import json
import re
from decimal import Decimal
from fractions import Fraction


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _ids(values, *, empty=False):
    require(isinstance(values, list) and (values or empty), "expected list of IDs")
    require(all(isinstance(v, str) and v.strip() for v in values), "invalid ID")
    require(len(values) == len(set(values)), "duplicate ID")
    return set(values)


def _number(text):
    # Bounded decimal strings keep unit conversions and tolerance edges exact.
    require(isinstance(text, str) and len(text) <= 64 and re.fullmatch(
        r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[Ee][+-]?[0-9]{1,3})?", text
    ), "expected finite decimal string")
    return Fraction(Decimal(text))


def _digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
            "expected SHA-256 digest")


def validate_key(key):
    """Validate private rule data before responses are parsed or scored."""
    require(isinstance(key, dict), "expected key object")
    kind = key.get("kind")
    fields = {
        "multiple_choice": {"options", "answer_id"},
        "numeric": {"targets", "rtol"},
        "evidence": {"source_ids", "required", "support", "answer_id"},
        "citation": {"source_ids", "required", "support"},
        "abstention": {"answer_ids", "reason_ids"},
        "programming": {"suite_sha256", "profile_sha256", "tests_total"},
        "expert": set(),
    }
    require(isinstance(kind, str) and kind in fields, "unknown scoring kind")
    require(set(key) == {"kind"} | fields[kind], "unexpected or missing key fields")
    if kind == "multiple_choice":
        require(isinstance(key["answer_id"], str) and key["answer_id"] in _ids(key["options"]),
                "answer outside option set")
    elif kind == "numeric":
        require(_number(key["rtol"]) >= 0, "negative relative tolerance")
        require(isinstance(key["targets"], dict) and key["targets"], "missing targets")
        for unit, target in key["targets"].items():
            require(isinstance(unit, str) and unit and isinstance(target, dict)
                    and set(target) == {"value", "atol"}, "invalid unit target")
            _number(target["value"])
            require(_number(target["atol"]) >= 0, "negative absolute tolerance")
    elif kind in {"evidence", "citation"}:
        universe = _ids(key["source_ids"])
        required = _ids(key["required"])
        support = key["support"]
        require(isinstance(support, dict) and required <= set(support), "unknown required claim")
        for claim, alternatives in support.items():
            require(isinstance(claim, str) and claim.strip() and isinstance(alternatives, list)
                    and alternatives, "invalid claim or alternatives")
            for cites in alternatives:
                require(_ids(cites) <= universe, "unknown reference in key")
        if kind == "evidence":
            require(isinstance(key["answer_id"], str) and key["answer_id"].strip(), "invalid answer")
    elif kind == "abstention":
        answers = _ids(key["answer_ids"], empty=True)
        reasons = _ids(key["reason_ids"], empty=True)
        require(bool(answers) != bool(reasons), "key must specify answers or abstention reasons")
    elif kind == "programming":
        _digest(key["suite_sha256"])
        _digest(key["profile_sha256"])
        require(type(key["tests_total"]) is int and key["tests_total"] > 0, "no frozen test count")


def _result(errors=(), status="scored"):
    return {"status": status, "pass": int(not errors) if status == "scored" else None,
            "errors": sorted(set(errors))}


def validate_verdict(verdict):
    """Check the same pass/error semantics at score lock and at custodian analysis."""
    status = verdict.get("status")
    require(status in ("scored", "inference_failure", "pending_review", "sandbox_unavailable",
                       "scoring_error"), "unknown result status")
    require("pass" in verdict, "missing score field")
    passed = verdict["pass"]
    errors = verdict.get("errors")
    require(isinstance(errors, list) and all(isinstance(e, str) and e.strip() for e in errors)
            and len(errors) == len(set(errors)), "invalid error labels")
    if status == "scored":
        require(type(passed) is int and passed in (0, 1), "expected binary pass")
        require(passed == int(not errors), "score and hard-error labels disagree")
    else:
        require(passed is None, "unscored response has score")
    if status == "pending_review":
        require(not errors, "pending objective has a known hard error")


def _pairs(pairs):
    obj = {}
    for k, v in pairs:
        require(k not in obj, "duplicate JSON member")
        obj[k] = v
    return obj


def _fields(obj, names):
    require(isinstance(obj, dict) and set(obj) == set(names), "unexpected response fields")


def _string(obj, k):
    value = obj[k]
    require(isinstance(value, str) and value.strip(), "invalid string")
    return value


def _attribution(key, obj):
    evidence = key["kind"] == "evidence"
    _fields(obj, ["answer_id", "claims"] if evidence else ["claims"])
    errors = []
    if evidence and _string(obj, "answer_id") != key["answer_id"]:
        errors.append("wrong_answer")
    seen = set()
    require(isinstance(obj["claims"], list), "invalid claims")
    for claim in obj["claims"]:
        _fields(claim, ["id", "sources"])
        cid = _string(claim, "id")
        require(cid not in seen, "duplicate claim")
        seen.add(cid)
        cites = _ids(claim["sources"], empty=True)
        if cites - set(key["source_ids"]):
            errors.append("unknown_reference")
        if cid not in key["support"]:
            errors.append("unsupported_claim")
        elif not any(cites == set(alt) for alt in key["support"][cid]):
            errors.append("attribution_error")
        if not cites:
            errors.append("missing_citation")
    if set(key["required"]) - seen:
        errors.append("missing_required_claim")
    return errors


def _structured(key, obj):
    kind = key["kind"]
    if kind == "multiple_choice":
        _fields(obj, ["answer_id"])
        answer = _string(obj, "answer_id")
        if answer not in key["options"]:
            return ["invalid_option"]
        return [] if answer == key["answer_id"] else ["wrong_answer"]
    if kind == "numeric":
        _fields(obj, ["value", "unit"])
        observed = _number(obj["value"])
        target = key["targets"].get(_string(obj, "unit"))
        if target is None:
            return ["wrong_unit"]
        expected = _number(target["value"])
        tolerance = max(_number(target["atol"]), _number(key["rtol"]) * abs(expected))
        return [] if abs(observed - expected) <= tolerance else ["numeric_error"]
    if kind == "abstention":
        require(isinstance(obj, dict), "expected decision")
        if obj.get("decision") == "abstain":
            _fields(obj, ["decision", "reason_id"])
            reason = _string(obj, "reason_id")
            if key["answer_ids"]:
                return ["unnecessary_abstention"]
            return [] if reason in key["reason_ids"] else ["wrong_abstention_reason"]
        _fields(obj, ["decision", "answer_id"])
        require(obj["decision"] == "answer", "unknown decision")
        answer = _string(obj, "answer_id")
        if not key["answer_ids"]:
            return ["failure_to_abstain"]
        return [] if answer in key["answer_ids"] else ["wrong_answer"]
    return _attribution(key, obj)


def _programming(key, response, receipt):
    if receipt is None:
        return _result(status="sandbox_unavailable")
    # Caller must authenticate a receipt from the approved executor separately.
    _fields(receipt, ["response_sha256", "suite_sha256", "profile_sha256", "tests_passed", "outcome"])
    require(receipt["response_sha256"] == hashlib.sha256(response.encode("utf-8")).hexdigest()
            and receipt["suite_sha256"] == key["suite_sha256"]
            and receipt["profile_sha256"] == key["profile_sha256"], "sandbox receipt binding mismatch")
    outcome = receipt["outcome"]
    require(outcome in ("completed", "resource_limit", "candidate_error", "infra_error"), "unknown outcome")
    passed = receipt["tests_passed"]
    require(type(passed) is int and 0 <= passed <= key["tests_total"], "invalid pass count")
    if outcome == "infra_error":
        return _result(["sandbox_error"], status="sandbox_unavailable")
    if outcome != "completed":
        return _result([outcome])
    return _result([] if passed == key["tests_total"] else ["test_failure"])


def score(key, response, *, verified_sandbox_receipt=None):
    """Score a delivered visible answer, never raw prompt+completion or an inference failure."""
    validate_key(key)
    require(isinstance(response, str), "expected generated answer text")
    if not response.strip():
        return _result(["empty_answer"])
    if key["kind"] == "programming":
        return _programming(key, response, verified_sandbox_receipt)
    require(verified_sandbox_receipt is None, "unexpected receipt")
    if key["kind"] == "expert":
        return _result(status="pending_review")
    # A strict declared JSON contract avoids silently extracting a favorable substring.
    try:
        obj = json.loads(response, object_pairs_hook=_pairs)
        errors = _structured(key, obj)
    except (ValueError, RecursionError):
        errors = ["format_error"]
    return _result(errors)


def final_score(objective, *, dual_review, final_rating):
    """Combine the locked objective record with the separate blinded review decision."""
    validate_verdict(objective)
    require(type(dual_review) is bool, "invalid review requirement")
    status = objective["status"]
    require(dual_review or status != "pending_review", "expert result requires review")
    if final_rating is None:
        # A known failed conjunct fixes the endpoint even with a missing human rating.
        # Review coverage is measured against its separate assignment ledger.
        if dual_review and objective["pass"] == 1:
            return _result(status="pending_review")
        return dict(objective)
    require(dual_review and type(final_rating) is int and final_rating in range(4),
            "unexpected or invalid final rating")
    if status not in {"scored", "pending_review"}:
        return dict(objective)
    errors = list(objective["errors"])
    if final_rating < 2:
        errors.append("expert_criterion_failure")
    return _result(errors)
