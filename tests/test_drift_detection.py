"""Synthetic run history only; no model or Slack calls."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import unittest

from drift_detection import DriftConfig, detect_slow_drift
from html_report import render_report
from slack_alerts import build_slack_message
from thresholds import ThresholdConfig


class DriftTests(unittest.TestCase):
    def setUp(self):
        self.dataset = {"dataset_version": "synthetic_v1", "cases": [
            {"id": str(i), "input": "Refund please.", "expected_category": "billing", "ideal_summary": "A refund is requested."}
            for i in range(100)]}

    def runs(self, rates):
        result = []
        for index, passed in enumerate(rates):
            result.append({"run_id": f"synthetic_{index}", "dataset_version": "synthetic_v1",
                           "started_at": (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=index)).isoformat(),
                           "judge_rubric_version": "v1", "model": "fake-classifier", "results": [
                {"case_id": str(i), "error": None, "elapsed_seconds": 0.1,
                 "output": {"category": "billing", "summary": "A refund is requested."},
                 "scores": {"error": None, "summary_relevance": 5 if i < passed else 3,
                            "judge": {"status": "scored", "error": None, "model": "fake-judge"}}}
                for i in range(100)]})
        return result

    def detect(self, runs, **kwargs):
        return detect_slow_drift(runs[-1], self.dataset,
                                 history=[(r, self.dataset) for r in runs[:-1]], **kwargs)

    def test_gradual_decline_fires_without_individual_alerts(self):
        result = self.detect(self.runs([92, 91, 90, 89, 88, 87, 86]))
        self.assertEqual(result["status"], "slow_drift")
        self.assertEqual(result["severity"], "warning")
        self.assertAlmostEqual(result["moving_average"], 0.89)
        self.assertEqual(result["per_run_alerts"], [])
        self.assertTrue(result["slow_drift"])

    def test_threshold_boundary_and_recovery(self):
        result = self.detect(self.runs([93, 92, 91, 90, 89, 88, 87]))
        self.assertEqual(result["status"], "ok")
        runs = self.runs([86, 87, 88, 89, 90, 91, 92, 100])
        result = self.detect(runs)
        self.assertAlmostEqual(result["moving_average"], sum([87, 88, 89, 90, 91, 92, 100]) / 700)
        self.assertEqual(result["status"], "ok")

    def test_insufficient_history(self):
        result = self.detect(self.runs([80] * 6))
        self.assertIsNone(result["moving_average"])
        self.assertEqual(result["status"], "insufficient_history")
        self.assertFalse(result["slow_drift"])

    def test_custom_window_and_threshold(self):
        result = self.detect(self.runs([95, 94, 93]), config=DriftConfig(3, 0.95))
        self.assertTrue(result["slow_drift"])
        self.assertAlmostEqual(result["moving_average"], 0.94)

    def test_existing_per_run_alert_suppresses_slow_drift(self):
        result = self.detect(self.runs([95, 85, 85, 85, 85, 85, 85]))
        self.assertEqual(result["status"], "per_run_alert_present")
        self.assertEqual(result["per_run_alerts"][0]["severity"], "critical")
        self.assertFalse(result["slow_drift"])
        # Existing per-run thresholds are configurable for this check too.
        result = self.detect(self.runs([92, 91, 90, 89, 88, 87, 86]), thresholds=ThresholdConfig(0.5, 8))
        self.assertEqual(result["status"], "per_run_alert_present")

    def test_transition_into_first_window_run_is_checked(self):
        result = self.detect(self.runs([100, 80, 80, 80, 80, 80, 80, 80]))
        self.assertEqual(result["status"], "per_run_alert_present")

    def test_invalid_run_breaks_window_instead_of_skipping(self):
        for mutation in ("error", "judge", "cohort", "policy", "consolidated"):
            runs = self.runs([89] * 8)
            row = runs[3]["results"][0]
            if mutation == "error": row["error"] = {"type": "TimeoutError"}
            elif mutation == "judge": row["scores"]["judge"]["model"] = "other"
            elif mutation == "cohort": runs[3]["results"].pop()
            elif mutation == "policy": runs[3]["pass_policy"] = "different"
            else: runs[3]["live_test"] = {"report_kind": "consolidated_live_subset"}
            with self.subTest(mutation=mutation):
                result = self.detect(runs)
                self.assertEqual(result["available_consecutive_runs"], 4)
                self.assertIsNone(result["moving_average"])
                self.assertEqual(len(result["excluded_runs"]), 1)

    def test_reference_change_breaks_history_even_same_version(self):
        runs = self.runs([89] * 7)
        old_dataset = deepcopy(self.dataset)
        old_dataset["cases"][0]["ideal_summary"] = "Changed reference."
        history = [(r, old_dataset if i == 3 else self.dataset) for i, r in enumerate(runs[:-1])]
        result = detect_slow_drift(runs[-1], self.dataset, history=history)
        self.assertEqual(result["available_consecutive_runs"], 3)
        self.assertFalse(result["slow_drift"])

    def test_reject_duplicate_future_and_naive_timestamps(self):
        runs = self.runs([90] * 7)
        for history in ([(runs[-1], self.dataset)], [(runs[0], self.dataset)] * 2,
                        [(self.runs([90] * 8)[-1], self.dataset)]):
            with self.assertRaises(ValueError):
                detect_slow_drift(runs[-1], self.dataset, history=history)
        runs[0]["started_at"] = "2026-01-01T00:00:00"
        with self.assertRaises(ValueError):
            self.detect(runs)

    def test_duplicate_run_ids_rejected(self):
        runs = self.runs([90] * 7)
        runs[1]["run_id"] = runs[0]["run_id"]
        with self.assertRaises(ValueError):
            self.detect(runs)

    def test_invalid_configuration(self):
        for window, threshold in ((1, 0.9), (True, 0.9), (7, -1), (7, 1.1), (7, True), (7, float("nan"))):
            with self.subTest(window=window, threshold=threshold), self.assertRaises(ValueError):
                DriftConfig(window, threshold)

    def test_integrations_and_no_mutation(self):
        runs = self.runs([92, 91, 90, 89, 88, 87, 86])
        original = deepcopy(runs)
        history = [(r, self.dataset) for r in runs[:-1]]
        payload = build_slack_message(runs[-1], self.dataset, previous=runs[-2],
                                      previous_dataset=self.dataset, history=history)
        self.assertTrue(payload["text"].startswith("WARN"))
        self.assertIn("SLOW DRIFT", payload["text"])
        self.assertIn("89.0%", payload["text"])
        html = render_report(runs[-1], self.dataset, history=history)
        self.assertIn("slow_drift", html)
        self.assertIn("89.0%", html)
        json.dumps(self.detect(runs), allow_nan=False)
        self.assertEqual(runs, original)
