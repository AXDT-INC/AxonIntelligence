"""Paired pilot analysis of a custodian-joined, locked private roster. Standard library."""
from collections import Counter
from math import ceil, isfinite, log, sqrt
from statistics import NormalDist, mean, median

from score import validate_verdict


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _fraction(n, d):
    return n / d if d else None


def _cluster_margin(clusters, alpha, span, tails):
    """Hoeffding half-width for independent clusters, arbitrary dependence within."""
    sizes = list(clusters.values())
    return span * sqrt(log(tails / alpha) * sum(n*n for n in sizes) / 2) / sum(sizes)


def _paired(rows):
    n = len(rows)
    complete = Counter()
    all_clusters, scored_clusters = Counter(), Counter()
    lower = upper = 0
    for item, a, b in rows:
        family = item["source_family"]
        all_clusters[family] += 1
        x, y = a["pass"], b["pass"]
        lower += (0 if y is None else y) - (1 if x is None else x)
        upper += (1 if y is None else y) - (0 if x is None else x)
        if x is not None and y is not None:
            complete[x, y] += 1
            scored_clusters[family] += 1
    wins, losses = complete[0, 1], complete[1, 0]
    done = sum(complete.values())
    difference = _fraction(wins - losses, done)
    interval = None
    if done:
        margin = _cluster_margin(scored_clusters, 0.05, 2, 2)
        interval = [max(-1, difference - margin), min(1, difference + margin)]
    q_upper = (wins + losses + n - done) / n + _cluster_margin(all_clusters, 0.05, 1, 1)
    return {
        "assigned": n, "complete": done, "incomplete": n - done,
        "complete_source_families": len(scored_clusters), "source_families": len(all_clusters),
        "B_wins": wins, "B_losses": losses, "ties_both_pass": complete[1, 1],
        "ties_both_fail": complete[0, 0], "discordance": _fraction(wins + losses, done),
        "discordance_upper95_for_planning": min(1, q_upper),
        "B_minus_A": difference, "complete_cluster_ci95": interval,
        "all_assigned_missing_score_bounds": [lower / n, upper / n],
    }


def _arm_summary(records):
    statuses = Counter(r["status"] for r in records)
    errors = Counter(code for r in records if r["status"] == "scored" for code in r["errors"])
    operational = Counter(code for r in records if r["status"] != "scored" for code in r["errors"])
    scored = statuses["scored"]
    passed = sum(r["pass"] for r in records if r["pass"] is not None)
    trunc = [r["truncated"] for r in records if r.get("truncated") is not None]
    return {
        "assigned": len(records), "status_counts": dict(statuses), "scored": scored,
        "passed": passed, "pass_rate_among_scored": _fraction(passed, scored),
        "scored_error_counts": dict(errors),
        "scored_error_rates": {code: n / scored for code, n in errors.items()},
        "unscored_issue_counts": dict(operational),
        "inference_failure_rate_assigned": statuses["inference_failure"] / len(records),
        "truncation_known": len(trunc), "truncation_rate_when_known": _fraction(sum(trunc), len(trunc)),
    }


def _distribution(values):
    return {"mean": mean(values), "median": median(values),
            "p95_nearest_rank": sorted(values)[ceil(.95 * len(values)) - 1]}


def _efficiency(rows):
    comparisons = {}
    for name in ("new_tokens", "latency_s"):
        pairs = [(a[name], b[name]) for _, a, b in rows if
                 a.get(name) is not None and b.get(name) is not None]
        data = {"paired": len(pairs)}
        if pairs:
            a_values, b_values = zip(*pairs)
            data.update(A=_distribution(a_values), B=_distribution(b_values),
                        paired_B_minus_A=_distribution([b-a for a, b in pairs]))
            if name == "new_tokens":
                data["B_over_A_total_tokens"] = _fraction(sum(b_values), sum(a_values))
        comparisons[name] = data
    pairs = [(a, b) for _, a, b in rows if all(r.get(k) is not None for r in (a, b)
             for k in ("new_tokens", "latency_s"))]
    comparisons["tokens_per_generation_second"] = {"paired": len(pairs)}
    if pairs:
        for arm, records in zip(("A", "B"), zip(*pairs)):
            comparisons["tokens_per_generation_second"][arm] = (
                sum(r["new_tokens"] for r in records) / sum(r["latency_s"] for r in records))
    return comparisons


def _check_result(row):
    validate_verdict(row)
    tokens = row.get("new_tokens")
    latency = row.get("latency_s")
    truncated = row.get("truncated")
    if tokens is not None:
        _require(type(tokens) is int and tokens >= 0, "invalid token count")
    if latency is not None:
        _require(type(latency) in (float, int) and isfinite(latency) and latency > 0, "invalid latency")
    _require(truncated is None or type(truncated) is bool, "invalid truncation indicator")
    _require(row["status"] != "inference_failure" or (tokens is None and latency is None and truncated is None),
             "failed generation metrics belong in attempt receipts")


def _abstention(rows):
    report = {}
    categories = {"answerable": "unnecessary_abstention", "insufficient_evidence": "failure_to_abstain"}
    _require(all(item.get("answerability") in categories for item, _, _ in rows),
             "missing abstention-opportunity classification")
    for category, error in categories.items():
        subset = [row for row in rows if row[0]["answerability"] == category]
        statistics = {}
        for index, arm in enumerate(("A", "B"), 1):
            scored = [row[index] for row in subset if row[index]["pass"] is not None]
            decisions = [r for r in scored if not {"format_error", "empty_answer"}.intersection(r["errors"])]
            events = sum(error in r["errors"] for r in decisions)
            n = len(subset)
            statistics[arm] = {
                "assigned": n, "unscored": n - len(scored),
                "invalid_or_empty": len(scored) - len(decisions),
                "interpretable": len(decisions), "events": events,
                "rate_among_interpretable": _fraction(events, len(decisions)),
                "all_assigned_rate_bounds": [events / n, (events + n - len(decisions)) / n] if n else None,
            }
        report[category] = {"event": error, "arms": statistics}
    return report


def analyze(roster, results):
    """Each roster entry is assigned to both arms. Missing results remain in the denominator."""
    items = {}
    for item in roster:
        _require(all(isinstance(item.get(k), str) and item[k].strip()
                     for k in ("item_id", "family", "source_family")), "invalid roster")
        _require(item["item_id"] not in items, "duplicate assigned item")
        items[item["item_id"]] = item
    _require(items, "empty assignment roster")
    observed = {}
    for row in results:
        rid = (row["item_id"], row["arm"])
        _require(rid[0] in items and rid[1] in ("A", "B"), "unassigned result")
        _require(rid not in observed, "duplicate arm result")
        _check_result(row)
        observed[rid] = row
    missing = {"status": "missing_result", "pass": None, "errors": []}
    rows = [(item, observed.get((i, "A"), missing), observed.get((i, "B"), missing))
            for i, item in items.items()]
    families = sorted({r[0]["family"] for r in rows})
    return {
        "overall": _paired(rows),
        "families": {f: _paired([r for r in rows if r[0]["family"] == f]) for f in families},
        "arms": {arm: _arm_summary([r[index] for r in rows]) for index, arm in enumerate(("A", "B"), 1)},
        "efficiency": _efficiency(rows),
        "abstention_opportunities": _abstention([r for r in rows if r[0]["family"] == "uncertainty_abstention"]),
    }


def agreement(pairs, *, levels=4):
    """Paired ordinal ratings in stable reviewer order; use None for every missing rating."""
    _require(type(levels) is int and 2 <= levels <= 10, "invalid scale")
    counts = [[0] * levels for _ in range(levels)]
    assigned = n = 0
    for left, right in pairs:
        assigned += 1
        _require(all(x is None or (type(x) is int and 0 <= x < levels) for x in (left, right)),
                 "out-of-scale rating")
        if left is not None and right is not None:
            counts[left][right] += 1
            n += 1
    report = {"assigned": assigned, "double_rated": n, "table": counts,
              "exact_agreement": None, "kappa": None, "linear_weighted_kappa": None}
    if n:
        a = [sum(row) for row in counts]
        b = [sum(col) for col in zip(*counts)]
        report["exact_agreement"] = sum(counts[i][i] for i in range(levels)) / n
        for name in ("kappa", "linear_weighted_kappa"):
            observed = expected = 0
            for i, row in enumerate(counts):
                for j, cell in enumerate(row):
                    distance = int(i != j) if name == "kappa" else abs(i-j)
                    observed += distance * cell * n
                    expected += distance * a[i] * b[j]
            report[name] = 1 - observed / expected if expected else None
    return report


def plan_pairs(discordance, target_difference, *, alpha=.05, power=.8, attrition=.0):
    """Approximate enrollment for independent pairs with one frozen binary endpoint."""
    _require(0 < target_difference <= discordance <= 1 and 0 < alpha < .5
             and .5 < power < 1 and 0 <= attrition < 1, "invalid planning assumptions")
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_power = NormalDist().inv_cdf(power)
    needed = ceil(((z_alpha * sqrt(discordance) + z_power * sqrt(discordance - target_difference**2))
                   / target_difference)**2)
    return {"complete_independent_pairs": needed, "enroll": ceil(needed / (1 - attrition))}
