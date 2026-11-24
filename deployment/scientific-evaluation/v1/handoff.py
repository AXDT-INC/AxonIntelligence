"""Post-lock custodian projection. Call only on authenticated and audited private records."""
import json
from pathlib import Path

from jsonschema import Draft7Validator

from runner import FAILURES
from score import validate_verdict


OUTCOMES = Draft7Validator(json.loads(Path(__file__).with_name("outcome.schema.json").read_text()))


def score_binding(receipt):
    """Private binding to include inside the authenticated anonymous score envelope.

    Custody adds this on ingestion. Never place a stable worker slot ticket in a reviewer packet.
    """
    return {k: receipt[k] for k in ("run_id", "item_id", "slot_ticket", "output_sha256")}


def analysis_row(receipt, frozen_score, arm):
    """After unblinding, validate the score's binding to the exact audited worker response."""
    if arm not in {"A", "B"}:
        raise ValueError("unrecognized arm")
    row = {"item_id": receipt["item_id"], "arm": arm}
    status = receipt["status"]
    if status in FAILURES - {"empty_output"}:
        if frozen_score is not None:
            raise ValueError("score supplied for undelivered response")
        row.update({"status": "inference_failure", "pass": None, "errors": [status]})
    else:
        if status not in {"ok", "empty_output", "length_limit"}:
            raise ValueError("unrecognized worker outcome")
        if frozen_score is None or frozen_score["binding"] != score_binding(receipt):
            raise ValueError("score does not bind this response")
        value = frozen_score["verdict"]
        invalid_item = value["status"] == "scoring_error" and "invalid_item" in value["errors"]
        if status == "empty_output" and not invalid_item:
            if value["status"] != "scored" or value["pass"] != 0 or "empty_answer" not in value["errors"]:
                raise ValueError("inconsistent blank-response score")
        row.update({k: value[k] for k in ("status", "pass", "errors")})
        row.update(new_tokens=receipt["output_tokens"], latency_s=receipt["measurement"]["latency_s"],
                   truncated=receipt["stop_reason"] == "length_limit")
    validate_verdict(row)
    if not OUTCOMES.is_valid(row):
        raise ValueError("invalid analysis projection")
    return row


def analysis_roster(items):
    roster = []
    for row in items:
        item = {"item_id": row["item_id"], "family": row["task_family"],
                "source_family": row["source_family_id"]}
        if item["family"] == "uncertainty_abstention":
            item["answerability"] = row["answerability"]
        roster.append(item)
    return roster
