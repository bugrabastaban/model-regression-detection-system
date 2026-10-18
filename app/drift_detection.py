"""Offline moving-average checks over consecutive, comparable scored runs."""

from dataclasses import asdict, dataclass
from datetime import datetime
import math
from statistics import mean

from app.comparison import PASS_POLICY, _index, _state, compare_runs
from app.thresholds import ThresholdConfig


@dataclass(frozen=True)
class DriftConfig:
    window_size: int = 7
    minimum_average_pass_rate: float = 0.90

    def __post_init__(self):
        if type(self.window_size) is not int or self.window_size < 2:
            raise ValueError("window_size must be an integer >= 2.")
        value = self.minimum_average_pass_rate
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("minimum_average_pass_rate must be a finite rate between 0 and 1.")


def _timestamp(report):
    try:
        stamp = datetime.fromisoformat(report["started_at"])
        if stamp.tzinfo is None:
            raise ValueError
        return stamp
    except (KeyError, ValueError, TypeError):
        raise ValueError("Each run requires a timezone-aware ISO 8601 started_at.") from None


def detect_slow_drift(current: dict, dataset: dict, *, history: list | None = None,
                      config: DriftConfig | None = None,
                      thresholds: ThresholdConfig | None = None) -> dict:
    """Evaluate the latest window, including current; no I/O or provider calls.

    history contains (scored_report, dataset_snapshot) pairs, excluding current.
    Entries are sorted chronologically. Duplicate timestamps/run IDs and future
    history are rejected. Incompatible or incomplete runs break the window;
    they are not silently skipped or counted as zero. Consolidated artifacts
    are not genuine runs and break the window too. Classifier model/prompt may
    change; references, cohort, pass policy, rubric and judge model must match.
    """
    config = config or DriftConfig()
    thresholds = thresholds or ThresholdConfig()
    if not isinstance(config, DriftConfig) or not isinstance(thresholds, ThresholdConfig):
        raise TypeError("Expected DriftConfig and ThresholdConfig.")
    _index(current, dataset)
    current_time = _timestamp(current)
    entries = list(history or []) + [(current, dataset)]
    times, run_ids = set(), set()
    for report, snapshot in entries:
        stamp = _timestamp(report)
        if stamp in times:
            raise ValueError("Duplicate run timestamps; do not supply current in history.")
        if stamp > current_time:
            raise ValueError("History cannot contain a run newer than current.")
        times.add(stamp)
        run_id = report.get("run_id")
        if run_id is not None:
            if not isinstance(run_id, str) or not run_id.strip() or run_id in run_ids:
                raise ValueError("Run IDs, when supplied, must be non-empty and unique.")
            run_ids.add(run_id)
        _index(report, snapshot)
    entries.sort(key=lambda pair: _timestamp(pair[0]))
    current_ids = {row["case_id"] for row in current["results"]}
    points, excluded = [], []
    streak = []
    latest_window = []
    for report, snapshot in entries:
        reasons = []
        if report.get("live_test", {}).get("report_kind") == "consolidated_live_subset":
            reasons.append("consolidated_artifact")
        if not current_ids or {row["case_id"] for row in report["results"]} != current_ids:
            reasons.append("different_or_empty_cohort")
        if report.get("pass_policy", PASS_POLICY) != PASS_POLICY or current.get("pass_policy", PASS_POLICY) != PASS_POLICY:
            reasons.append("incompatible_pass_policy")
        diff = compare_runs(current, report, current_dataset=dataset, previous_dataset=snapshot, thresholds=thresholds)
        for key in ("changed_cases", "unevaluable_cases", "judge_incompatible_cases"):
            if diff[key]:
                reasons.append(key)
        rows, refs = _index(report, snapshot)
        states = [_state(row, refs[key]) for key, row in rows.items()]
        if not states or any(state["pass"] is None for state in states):
            reasons.append("incomplete_scores")
        point = {"run_id": report.get("run_id"), "started_at": report["started_at"],
                 "pass_rate": None, "moving_average": None, "case_count": len(rows),
                 "exclusion_reasons": reasons}
        if reasons:
            streak = []
            latest_window = []
            excluded.append({"started_at": report["started_at"], "reasons": reasons})
        else:
            point["pass_rate"] = mean(state["pass"] for state in states)
            streak.append((report, snapshot, point))
            if len(streak) >= config.window_size:
                latest_window = streak[-config.window_size:]
                point["moving_average"] = mean(item[2]["pass_rate"] for item in latest_window)
        points.append(point)
    average = points[-1]["moving_average"]
    result = {"metric": "overall_pass_rate", "pass_policy": PASS_POLICY,
              "config": asdict(config), "status": "insufficient_history",
              "severity": "not_evaluated", "slow_drift": False,
              "moving_average": average, "available_consecutive_runs": len(streak),
              "window_runs": [item[0]["started_at"] for item in latest_window],
              "per_run_alerts": [], "excluded_runs": excluded, "series": points}
    if average is None:
        return result
    # Check every available transition into/within the latest window. If an
    # immediately preceding genuine run is available, include its transition.
    start = len(entries) - config.window_size
    for index in range(max(1, start), len(entries)):
        if points[index - 1]["exclusion_reasons"]:
            continue
        old, old_dataset = entries[index - 1]
        new, new_dataset = entries[index]
        diff = compare_runs(new, old, current_dataset=new_dataset, previous_dataset=old_dataset, thresholds=thresholds)
        level = diff["threshold_assessment"]["severity"]
        if level in ("warning", "critical"):
            result["per_run_alerts"].append({"started_at": new["started_at"], "severity": level})
    below = average < config.minimum_average_pass_rate and not math.isclose(
        average, config.minimum_average_pass_rate, rel_tol=0, abs_tol=1e-12)
    if result["per_run_alerts"]:
        result.update(status="per_run_alert_present", severity="not_evaluated")
    elif below:
        result.update(status="slow_drift", severity="warning", slow_drift=True)
    else:
        result.update(status="ok", severity="ok")
    return result
