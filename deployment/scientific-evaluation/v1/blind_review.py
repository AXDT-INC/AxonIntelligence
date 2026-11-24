"""Custodian-side single-response blinding. No filesystem, model or network access."""
import secrets
from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    # response_id is a custodian-generated opaque handle, with no model identity.
    response_id: str
    question: str
    source_packet: str
    visible_rubric: str
    response: str


@dataclass(frozen=True)
class Packet:
    ticket: str
    question: str
    source_packet: str
    visible_rubric: str
    response: str


@dataclass(frozen=True)
class Rating:
    ticket: str
    rating: int
    rationale: str
    fatal_error: bool


def prepare(cases, reviewer_ids):
    """Create private per-reviewer deliveries and a separate custodian-only ledger."""
    if len(reviewer_ids) != 2 or len(set(reviewer_ids)) != 2 or not all(reviewer_ids):
        raise ValueError("two distinct assigned reviewers required")
    if not cases or len({case.response_id for case in cases}) != len(cases):
        raise ValueError("expected a nonempty unique response roster")
    deliveries = {reviewer: [] for reviewer in reviewer_ids}
    ledger = {}
    for reviewer, delivery in deliveries.items():
        for case in cases:
            ticket = secrets.token_hex(16)
            if ticket in ledger:
                raise RuntimeError("ticket collision")
            delivery.append(Packet(ticket, case.question, case.source_packet, case.visible_rubric,
                                   case.response))
            ledger[ticket] = (case.response_id, reviewer)
        secrets.SystemRandom().shuffle(delivery)
    return deliveries, ledger


def value(record):
    if type(record.rating) is not int or record.rating not in range(4):
        raise ValueError("rating must be an integer from 0 to 3")
    if type(record.fatal_error) is not bool or (record.fatal_error and record.rating != 0):
        raise ValueError("a fatal scientific error receives the lowest anchor")
    if not isinstance(record.rationale, str) or not record.rationale.strip():
        raise ValueError("evidence or rationale required")
    return record.rating


def lock_ratings(ledger, submissions):
    """Input (authenticated_reviewer_id, Rating) pairs from independently sealed deliveries."""
    locked = {}
    for reviewer, record in submissions:
        if record.ticket in locked or record.ticket not in ledger or reviewer != ledger[record.ticket][1]:
            raise ValueError("duplicate, unknown or misassigned rating")
        value(record)
        locked[record.ticket] = record
    if locked.keys() != ledger.keys():
        raise ValueError("incomplete independent ratings")
    return locked


def finalize(ledger, locked, adjudications):
    """Adjudications are separate authenticated (third_reviewer, Rating) per disputed response.

    Here an adjudicator's ticket is the opaque response handle. The custodian supplies only
    that response, its approved evidence and rubric, and anonymized reasons to this reviewer.
    """
    if ledger.keys() != locked.keys():
        raise ValueError("incomplete independent lock")
    pairs = {}
    for ticket, (response_id, reviewer) in ledger.items():
        record = locked[ticket]
        if record.ticket != ticket:
            raise ValueError("ticket changed")
        pairs.setdefault(response_id, {})[reviewer] = (value(record), record.fatal_error)
    disputed = set()
    final = {}
    for response_id, ratings in pairs.items():
        if len(ratings) != 2:
            raise ValueError("two independent ratings required")
        if len(set(ratings.values())) == 1:
            final[response_id] = next(iter(ratings.values()))[0]
        else:
            disputed.add(response_id)
    if set(adjudications) != disputed:
        raise ValueError("missing or unexpected adjudication")
    for response_id, (reviewer, record) in adjudications.items():
        if not reviewer or reviewer in pairs[response_id] or record.ticket != response_id:
            raise ValueError("adjudicator conflict or wrong response")
        final[response_id] = value(record)
    return final
