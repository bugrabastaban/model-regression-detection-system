"""Offline comparison of scored runs over unchanged reference cases."""

from app.contracts import ClassifierOutput
from app.thresholds import ThresholdConfig, assess_comparison
from pydantic import ValidationError
from app.token_usage import TokenUsage


PASS_POLICY = "category_match_and_summary_at_least_4_v1"
CATEGORIES = ("billing", "technical", "account", "general")
REFERENCE_FIELDS = ("input", "expected_category", "ideal_summary")


def _token_counts(row: dict, provider: str) -> dict:
    """Read optional scored counters, without changing quality eligibility.

    Scored Gemini fields are authoritative; runner fields are a legacy fallback
    only when the scored token_usage field is absent. Invalid usage is unknown.
    Judge output errors alone do not discard already captured token usage.
    """
    scores = row["scores"]
    if provider == "gemini":
        source = scores if "token_usage" in scores else row
        invalid = source.get("token_usage_error") is not None or row.get("token_usage_error") is not None
    else:
        source = scores.get("judge") or {}
        invalid = source.get("token_usage_error") is not None
    if invalid:
        return TokenUsage().model_dump()
    raw = source.get("token_usage")
    try:
        return TokenUsage.model_validate({} if raw is None else raw).model_dump()
    except ValidationError:
        return TokenUsage().model_dump()


def _token_comparison(previous: dict, current: dict) -> dict:
    """Per-provider counter deltas, in tokens; no totals or costs are inferred."""
    result = {}
    for provider in ("gemini", "groq"):
        before, after = _token_counts(previous, provider), _token_counts(current, provider)
        result[provider] = {"previous": before, "current": after, "delta_tokens": {
            key: after[key] - before[key] if before[key] is not None and after[key] is not None else None
            for key in before}}
    return result


def _index(report: dict, dataset: dict) -> tuple[dict, dict]:
    if report.get("dataset_version") != dataset.get("dataset_version"):
        raise ValueError("Report and reference dataset versions must match.")
    references = {}
    for case in dataset["cases"]:
        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id.strip() or case_id in references:
            raise ValueError("Dataset IDs must be non-empty and unique.")
        if not isinstance(case["input"], str) or not case["input"].strip():
            raise ValueError("Reference input must be non-empty.")
        ClassifierOutput(category=case["expected_category"], summary=case["ideal_summary"])
        references[case_id] = case
    rows = {}
    for row in report["results"]:
        case_id = row["case_id"]
        if case_id in rows or case_id not in references:
            raise ValueError("Result IDs must be unique and present in the dataset.")
        if not isinstance(row.get("scores"), dict):
            raise ValueError("Comparison requires scored results.")
        rows[case_id] = row
    return rows, references


def _state(row: dict, reference: dict) -> dict:
    scores = row["scores"]
    accuracy = None
    if row.get("error") is None and scores.get("error") is None:
        try:
            output = ClassifierOutput.model_validate(row.get("output"))
            accuracy = int(output.category == reference["expected_category"])
        except ValueError:
            pass
    judge = scores.get("judge") or {}
    rating = scores.get("summary_relevance")
    valid_judge = (
        judge.get("status") == "scored" and judge.get("error") is None
        and type(rating) is int and 1 <= rating <= 5
        and isinstance(judge.get("model"), str) and bool(judge["model"].strip())
    )
    passed = None if accuracy is None or not valid_judge else accuracy == 1 and rating >= 4
    return {"category_match": accuracy, "pass": passed,
            "summary_relevance": rating if valid_judge else None,
            "judge_model": judge.get("model")}


def _rates(previous: list, current: list) -> dict:
    count = len(previous)
    old = sum(previous) / count if count else None
    new = sum(current) / count if count else None
    return {"compared_cases": count, "previous_rate": old, "current_rate": new,
            "delta_percentage_points": (sum(current) - sum(previous)) / count * 100 if count else None}


def _metadata(report: dict | None) -> dict | None:
    if report is None:
        return None
    return {key: report.get(key) for key in
            ("run_id", "started_at", "dataset_version", "prompt_version", "model", "judge_rubric_version")}


def compare_runs(
    current: dict,
    previous: dict | None = None,
    *,
    current_dataset: dict,
    previous_dataset: dict | None = None,
    thresholds: ThresholdConfig | None = None,
) -> dict:
    """Compare explicit current/previous reports, without model calls or I/O.

    Supply each run's actual dataset snapshot, even if version names coincide.
    Never infer the prior run from filenames. IDs are matched independently of
    order. Only unchanged input/category/reference-summary cases enter deltas.
    Overall pass needs category correctness and judge >=4; errors/missing judge
    are unevaluable. Both runs must be evaluable on the same cases. Category
    accuracy has its own paired denominator and does not require a judge.
    Different/unknown judge rubric or model excludes a pair from pass deltas.
    Legacy reports without pass_policy use PASS_POLICY. Explicit policies must
    equal the supported policy; otherwise no deltas are calculated.
    token_usage_comparisons contains unchanged common cases, independently of
    quality eligibility. Provider counters/deltas never enter quality decisions.
    """
    if thresholds is None:
        thresholds = ThresholdConfig()
    if not isinstance(thresholds, ThresholdConfig):
        raise TypeError("thresholds must be a ThresholdConfig.")
    now, now_refs = _index(current, current_dataset)
    result = {
        "status": "baseline" if previous is None else "compared",
        "pass_policy": PASS_POLICY,
        "current_run": _metadata(current), "previous_run": _metadata(previous),
        "current_case_count": len(now), "previous_case_count": 0,
        "added_cases": [], "removed_cases": [], "changed_cases": [],
        "unevaluable_cases": [], "judge_incompatible_cases": [],
        "regressions": [], "improvements": [], "case_changes": [],
        "token_usage_comparisons": [],
        "overall_pass_rate": _rates([], []),
        "per_category_accuracy": {category: _rates([], []) for category in CATEGORIES},
    }
    policies = {"current": current.get("pass_policy", PASS_POLICY),
                "previous": previous.get("pass_policy", PASS_POLICY) if previous is not None else None}
    compatible = policies["current"] == PASS_POLICY and (
        previous is None or policies["previous"] == PASS_POLICY)
    result["pass_policy_compatibility"] = {"compatible": compatible, **policies}
    if not compatible:
        result["status"] = "incompatible_pass_policy"
        if previous is not None:
            if previous_dataset is None:
                raise ValueError("The previous run's dataset snapshot is required.")
            old, _ = _index(previous, previous_dataset)
            result["previous_case_count"] = len(old)
        result["coverage"] = {"unchanged_common_cases": 0, "pass_comparable_cases": 0,
                              "category_comparable_cases": 0}
        result["threshold_assessment"] = assess_comparison(result, thresholds)
        return result
    if previous is None:
        result["threshold_assessment"] = assess_comparison(result, thresholds)
        return result
    if previous_dataset is None:
        raise ValueError("The previous run's dataset snapshot is required.")
    old, old_refs = _index(previous, previous_dataset)
    result["previous_case_count"] = len(old)
    result["added_cases"] = [key for key in now if key not in old]
    result["removed_cases"] = [key for key in old if key not in now]
    pass_old, pass_now = [], []
    category_pairs = {category: ([], []) for category in CATEGORIES}
    for case_id in now:
        if case_id not in old:
            continue
        changed = [field for field in REFERENCE_FIELDS
                   if old_refs[case_id][field] != now_refs[case_id][field]]
        if changed:
            result["changed_cases"].append({"case_id": case_id, "fields": changed})
            continue
        result["token_usage_comparisons"].append({
            "case_id": case_id, **_token_comparison(old[case_id], now[case_id])})
        before = _state(old[case_id], old_refs[case_id])
        after = _state(now[case_id], now_refs[case_id])
        category = now_refs[case_id]["expected_category"]
        if before["category_match"] is not None and after["category_match"] is not None:
            category_pairs[category][0].append(before["category_match"])
            category_pairs[category][1].append(after["category_match"])
        entry = {"case_id": case_id, "expected_category": category,
                 "previous": before, "current": after}
        result["case_changes"].append(entry)
        if before["pass"] is None or after["pass"] is None:
            result["unevaluable_cases"].append(case_id)
            continue
        rubric = current.get("judge_rubric_version")
        if not isinstance(rubric, str) or not rubric.strip() or rubric != previous.get("judge_rubric_version") or before["judge_model"] != after["judge_model"]:
            result["judge_incompatible_cases"].append(case_id)
            continue
        pass_old.append(before["pass"])
        pass_now.append(after["pass"])
        if before["pass"] and not after["pass"]:
            result["regressions"].append(case_id)
        elif not before["pass"] and after["pass"]:
            result["improvements"].append(case_id)
    result["overall_pass_rate"] = _rates(pass_old, pass_now)
    result["per_category_accuracy"] = {
        category: _rates(*pairs) for category, pairs in category_pairs.items()
    }
    result["coverage"] = {
        "unchanged_common_cases": len(result["case_changes"]),
        "pass_comparable_cases": len(pass_old),
        "category_comparable_cases": sum(len(pair[0]) for pair in category_pairs.values()),
    }
    result["threshold_assessment"] = assess_comparison(result, thresholds)
    return result
