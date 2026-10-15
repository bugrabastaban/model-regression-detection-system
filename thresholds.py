"""Configurable regression thresholds; these are not hypothesis tests."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ThresholdConfig:
    """Drop thresholds in percentage points, using strict greater-than."""

    warning_percentage_points: float = 3.0
    critical_percentage_points: float = 8.0

    def __post_init__(self):
        values = (self.warning_percentage_points, self.critical_percentage_points)
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise ValueError("Thresholds must be finite numbers.")
        if not 0 <= values[0] < values[1] <= 100:
            raise ValueError("Thresholds must satisfy 0 <= warning < critical <= 100.")


def classify_delta(delta_percentage_points: float | None, config: ThresholdConfig) -> str:
    """Flag drops only. Positive deltas are improvements, not regressions."""
    if delta_percentage_points is None:
        return "not_evaluated"
    if type(delta_percentage_points) not in (int, float) or not math.isfinite(delta_percentage_points):
        raise ValueError("Delta must be finite or None.")
    drop = -delta_percentage_points
    # Avoid floating arithmetic turning an exact boundary into an exceedance.
    def exceeds(threshold):
        return drop > threshold and not math.isclose(drop, threshold, rel_tol=0, abs_tol=1e-10)
    if exceeds(config.critical_percentage_points):
        return "critical"
    if exceeds(config.warning_percentage_points):
        return "warning"
    return "ok"


def assess_comparison(comparison: dict, config: ThresholdConfig) -> dict:
    """Assess overall and per-category metrics without mutating the comparison."""
    overall = comparison["overall_pass_rate"]
    categories = comparison["per_category_accuracy"]
    def assess(metric):
        return {"severity": classify_delta(metric["delta_percentage_points"], config),
                "delta_percentage_points": metric["delta_percentage_points"],
                "compared_cases": metric["compared_cases"]}
    assessments = {"overall_pass_rate": assess(overall),
                   "per_category_accuracy": {key: assess(value) for key, value in categories.items()}}
    levels = [assessments["overall_pass_rate"]["severity"],
              *(value["severity"] for value in assessments["per_category_accuracy"].values())]
    severity = next((level for level in ("critical", "warning", "ok") if level in levels), "not_evaluated")
    return {"method": "drop_thresholds", "statistical_test": False,
            "thresholds": asdict(config), "severity": severity, **assessments}
