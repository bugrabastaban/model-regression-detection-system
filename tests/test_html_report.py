"""Offline HTML rendering, escaping, baseline and regression tests."""

from copy import deepcopy
from html.parser import HTMLParser
import json
from pathlib import Path
import unittest

from html_report import render_report


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.dataset = {"dataset_version": "v1", "cases": [
            {"id": "a", "input": "Refund please.", "expected_category": "billing", "ideal_summary": "A refund is requested."}]}
        self.report = {"dataset_version": "v1", "prompt_version": "v1", "model": "test-model",
                       "started_at": "2026-10-15T12:00:00+00:00", "judge_rubric_version": "v1", "results": [
            {"case_id": "a", "output": {"category": "billing", "summary": "A refund is requested."},
             "error": None, "elapsed_seconds": 1.2,
             "scores": {"summary_relevance": 5, "judge": {"status": "scored", "model": "judge", "reason": "Correct."}}}]}

    def test_baseline_and_single_point_are_honest(self):
        html = render_report(self.report, self.dataset)
        self.assertIn("Baseline yok", html)
        self.assertIn("100.0%", html)
        self.assertIn("Tek sonuç kaydı var", html)
        self.assertIn("Karşılaştırma verisi yok", html)
        self.assertIn("test-model", html)
        self.assertIn("PASS", html)

    def test_regressed_case_shows_old_and_new(self):
        current = deepcopy(self.report)
        current["started_at"] = "2026-10-15T13:00:00+00:00"
        current["results"][0]["output"]["summary"] = "The customer asks for a password reset."
        current["results"][0]["scores"]["summary_relevance"] = 1
        html = render_report(current, self.dataset, previous=self.report, previous_dataset=self.dataset)
        self.assertIn("Gerileme: 1", html)
        self.assertIn("A refund is requested.", html)
        self.assertIn("The customer asks for a password reset.", html)
        self.assertIn("-100.00", html)
        self.assertIn("Son 2 karşılaştırılabilir", html)

    def test_html_in_data_is_escaped(self):
        attack = '<script>alert("x")</script>'
        self.dataset["cases"][0]["input"] = attack
        self.report["model"] = attack
        self.report["results"][0]["output"]["summary"] = attack
        html = render_report(self.report, self.dataset)
        self.assertNotIn(attack, html)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn('<script', html)

    def test_history_limits_and_changed_cohort_excluded(self):
        old = deepcopy(self.report)
        old["started_at"] = "2026-10-14T12:00:00+00:00"
        changed = deepcopy(self.dataset)
        changed["cases"][0]["input"] = "Changed email"
        html = render_report(self.report, self.dataset, history=[(old, changed)])
        self.assertIn("Tek sonuç kaydı var", html)
        html = render_report(self.report, self.dataset, history=[(old, self.dataset)], last_n=1)
        self.assertIn("Tek sonuç kaydı var", html)
        with self.assertRaises(ValueError):
            render_report(self.report, self.dataset, last_n=0)

    def test_missing_scores_not_rendered_as_success(self):
        self.report["results"][0]["scores"]["summary_relevance"] = None
        html = render_report(self.report, self.dataset)
        self.assertIn("DEĞERLENDİRİLEMEDİ", html)
        self.assertIn("0/0 değerlendirilebilir", html)

    def test_real_saved_subset_and_no_mutation(self):
        root = Path(__file__).resolve().parents[1]
        report = json.loads((root / 'test_results/combined_subset_live_20261015.json').read_text(encoding='utf-8'))
        dataset = json.loads((root / 'datasets/golden_v1.json').read_text(encoding='utf-8-sig'))
        before = deepcopy(report)
        html = render_report(report, dataset)
        HTMLParser().feed(html)
        self.assertIn("3 / 92 kayıt", html)
        self.assertIn("4.67/5", html)
        self.assertIn("tek kesintisiz çalışma değildir", html)
        self.assertIn("HTTP 503", html)
        self.assertEqual(report, before)
