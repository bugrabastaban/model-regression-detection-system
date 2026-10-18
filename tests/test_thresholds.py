"""Offline threshold boundaries and comparison integration."""

from copy import deepcopy
import unittest

from app.comparison import compare_runs
from app.thresholds import ThresholdConfig, assess_comparison, classify_delta


class ThresholdTests(unittest.TestCase):
    def test_default_boundaries_and_improvements(self):
        config = ThresholdConfig()
        for delta, expected in ((None, "not_evaluated"), (0, "ok"), (20, "ok"),
                                (-2.5, "ok"), (-3, "ok"), (-3.01, "warning"),
                                (-8, "warning"), (-8.01, "critical"), (-100, "critical")):
            with self.subTest(delta=delta):
                self.assertEqual(classify_delta(delta, config), expected)

    def test_custom_thresholds(self):
        config = ThresholdConfig(warning_percentage_points=1, critical_percentage_points=5)
        self.assertEqual(classify_delta(-2, config), "warning")
        self.assertEqual(classify_delta(-6, config), "critical")
        self.assertEqual(classify_delta(-1, config), "ok")

    def test_reject_invalid_configuration(self):
        for warning, critical in ((-1, 8), (8, 3), (3, 3), (3, 101),
                                  (True, 8), ("3", 8), (float("nan"), 8), (3, float("inf"))):
            with self.subTest(warning=warning, critical=critical), self.assertRaises(ValueError):
                ThresholdConfig(warning, critical)

    def test_invalid_deltas(self):
        for delta in (True, "3", float("nan"), float("inf")):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                classify_delta(delta, ThresholdConfig())

    def test_rounding_at_boundary(self):
        self.assertEqual(classify_delta((0.97 - 1) * 100, ThresholdConfig()), "ok")
        self.assertEqual(classify_delta((0.92 - 1) * 100, ThresholdConfig()), "warning")

    def test_worst_metric_and_no_mutation(self):
        comparison = {"overall_pass_rate": {"delta_percentage_points": 0, "compared_cases": 80},
                      "per_category_accuracy": {"billing": {"delta_percentage_points": -10, "compared_cases": 20},
                                                "general": {"delta_percentage_points": None, "compared_cases": 0}}}
        before = deepcopy(comparison)
        result = assess_comparison(comparison, ThresholdConfig())
        self.assertEqual(result["severity"], "critical")
        self.assertEqual(result["overall_pass_rate"]["severity"], "ok")
        self.assertEqual(result["per_category_accuracy"]["general"]["severity"], "not_evaluated")
        self.assertFalse(result["statistical_test"])
        self.assertEqual(comparison, before)

    def test_integration_two_of_80_flips_and_config_override(self):
        cases = [{"id": str(i), "input": "Refund", "expected_category": "billing", "ideal_summary": "Refund requested."}
                 for i in range(80)]
        dataset = {"dataset_version": "v1", "cases": cases}
        previous = {"dataset_version": "v1", "judge_rubric_version": "v1", "results": [
            {"case_id": str(i), "error": None, "output": {"category": "billing", "summary": "Refund requested."},
             "scores": {"summary_relevance": 5, "judge": {"status": "scored", "model": "judge"}}}
            for i in range(80)]}
        current = deepcopy(previous)
        for row in current["results"][:2]:
            row["output"]["category"] = "general"
        result = compare_runs(current, previous, current_dataset=dataset, previous_dataset=dataset)
        self.assertEqual(result["overall_pass_rate"]["delta_percentage_points"], -2.5)
        self.assertEqual(result["threshold_assessment"]["severity"], "ok")
        custom = compare_runs(current, previous, current_dataset=dataset, previous_dataset=dataset,
                              thresholds=ThresholdConfig(1, 2))
        self.assertEqual(custom["threshold_assessment"]["severity"], "critical")
        self.assertEqual(len(custom["regressions"]), 2)

    def test_baseline_is_not_evaluated(self):
        result = compare_runs({"dataset_version": "v1", "results": []},
                              current_dataset={"dataset_version": "v1", "cases": []})
        self.assertEqual(result["threshold_assessment"]["severity"], "not_evaluated")
