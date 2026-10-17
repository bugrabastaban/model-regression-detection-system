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

    def test_full_coverage_is_independent_of_pass_result(self):
        current = deepcopy(self.report)
        current["results"][0]["scores"]["summary_relevance"] = 1
        html = render_report(current, self.dataset, previous=self.report, previous_dataset=self.dataset)
        self.assertIn("İşlenen kapsam: Tam kapsam (1 / 1 kayıt)", html)
        self.assertIn("Değerlendirilebilir kapsam: Tam kapsam (1 / 1 kayıt)", html)
        self.assertIn("Baseline ile pass karşılaştırması: Tam kapsam (1 / 1 kayıt)", html)
        self.assertNotIn("Kısmi kapsam", html)
        self.assertIn("Gerileme: 1", html)

    def test_missing_records_make_coverage_partial_against_supplied_snapshot(self):
        extra = {**self.dataset["cases"][0], "id": "b"}
        self.dataset["cases"].append(extra)
        html = render_report(self.report, self.dataset, previous=self.report, previous_dataset=self.dataset)
        self.assertIn("İşlenen kapsam: Kısmi kapsam (1 / 2 kayıt)", html)
        self.assertIn("Değerlendirilebilir kapsam: Kısmi kapsam (1 / 2 kayıt)", html)
        self.assertIn("Baseline ile pass karşılaştırması: Kısmi kapsam (1 / 2 kayıt)", html)

    def test_unevaluable_records_do_not_reduce_processed_count(self):
        for failed_side in ("current", "previous"):
            with self.subTest(failed_side=failed_side):
                current, previous = deepcopy(self.report), deepcopy(self.report)
                failed = current if failed_side == "current" else previous
                failed["results"][0]["scores"]["summary_relevance"] = None
                html = render_report(current, self.dataset, previous=previous, previous_dataset=self.dataset)
                self.assertIn("İşlenen kapsam: Tam kapsam (1 / 1 kayıt)", html)
                evaluable = "Kısmi kapsam (0 / 1 kayıt)" if failed_side == "current" else "Tam kapsam (1 / 1 kayıt)"
                self.assertIn(f"Değerlendirilebilir kapsam: {evaluable}", html)
                self.assertIn("Baseline ile pass karşılaştırması: Kısmi kapsam (0 / 1 kayıt)", html)

    def test_full_current_coverage_without_baseline_does_not_claim_comparison(self):
        html = render_report(self.report, self.dataset)
        self.assertIn("Değerlendirilebilir kapsam: Tam kapsam (1 / 1 kayıt)", html)
        self.assertIn("Baseline ile pass karşılaştırması: Baseline yok; hesaplanmadı.", html)
        self.assertNotIn("Baseline ile pass karşılaştırması: Tam kapsam", html)

    def test_comparison_coverage_uses_existing_eligibility_rules(self):
        for reason in ("reference", "judge", "policy"):
            with self.subTest(reason=reason):
                previous, snapshot = deepcopy(self.report), deepcopy(self.dataset)
                if reason == "reference":
                    snapshot["cases"][0]["input"] = "Changed request."
                elif reason == "judge":
                    previous["results"][0]["scores"]["judge"]["model"] = "different-judge"
                else:
                    previous["pass_policy"] = "unsupported"
                html = render_report(self.report, self.dataset, previous=previous, previous_dataset=snapshot)
                if reason == "policy":
                    self.assertIn("Baseline ile pass karşılaştırması: Pass politikaları uyumsuz; hesaplanmadı.", html)
                else:
                    self.assertIn("Baseline ile pass karşılaştırması: Kısmi kapsam (0 / 1 kayıt)", html)
                self.assertIn("Değerlendirilebilir kapsam: Tam kapsam (1 / 1 kayıt)", html)

    def test_empty_data_does_not_claim_full_coverage(self):
        current = deepcopy(self.report)
        current["results"] = []
        for cases in ([], self.dataset["cases"]):
            with self.subTest(expected=len(cases)):
                dataset = {**self.dataset, "cases": cases}
                html = render_report(current, dataset, previous=current, previous_dataset=dataset)
                if cases:
                    self.assertIn("İşlenen kapsam: Kısmi kapsam (0 / 1 kayıt)", html)
                    self.assertIn("Değerlendirilebilir kapsam: Kısmi kapsam (0 / 1 kayıt)", html)
                    self.assertIn("Baseline ile pass karşılaştırması: Kısmi kapsam (0 / 1 kayıt)", html)
                else:
                    self.assertIn("İşlenen kapsam: Veri yok (0 / 0 kayıt)", html)
                    self.assertIn("Değerlendirilebilir kapsam: Veri yok (0 / 0 kayıt)", html)
                    self.assertIn("Baseline ile pass karşılaştırması: Veri yok (0 / 0 kayıt)", html)
                self.assertNotIn("Tam kapsam", html)

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

    def test_previous_copy_in_history_is_counted_once_with_or_without_run_id(self):
        for use_id in (False, True):
            with self.subTest(run_id=use_id):
                previous = deepcopy(self.report)
                previous["started_at"] = "2026-10-14T12:00:00+00:00"
                if use_id:
                    previous["run_id"] = "previous-run"
                history = [(deepcopy(previous), deepcopy(self.dataset))]
                before = deepcopy(history)
                html = render_report(self.report, self.dataset, previous=previous,
                                     previous_dataset=self.dataset, history=history)
                self.assertEqual(html.count("<circle "), 2)
                self.assertIn("Son 2 karşılaştırılabilir", html)
                self.assertEqual(history, before)

    def test_duplicate_history_and_current_copies_do_not_fill_trend_window(self):
        oldest = deepcopy(self.report)
        oldest["run_id"] = "oldest"
        oldest["started_at"] = "2026-10-13T12:00:00+00:00"
        previous = deepcopy(self.report)
        previous["run_id"] = "previous"
        previous["started_at"] = "2026-10-14T12:00:00+00:00"
        self.report["run_id"] = "current"
        history = [(report, self.dataset) for report in (
            oldest, previous, deepcopy(previous), deepcopy(self.report))]
        html = render_report(self.report, self.dataset, previous=previous,
                             previous_dataset=self.dataset, history=history, last_n=3)
        self.assertEqual(html.count("<circle "), 3)
        self.assertIn(f'<li>{oldest["started_at"]}', html)
        self.assertIn("Son 3 karşılaştırılabilir", html)

    def test_run_id_takes_precedence_over_changed_metadata_in_saved_copy(self):
        previous = deepcopy(self.report)
        previous["run_id"] = "previous"
        previous["started_at"] = "2026-10-14T12:00:00+00:00"
        copied = deepcopy(previous)
        copied["prompt_timestamp"] = "additional metadata"
        html = render_report(self.report, self.dataset, previous=previous,
                             previous_dataset=self.dataset, history=[(copied, self.dataset)])
        self.assertEqual(html.count("<circle "), 2)

    def test_incompatible_pass_policy_is_visible_and_excluded_from_trend(self):
        previous = deepcopy(self.report)
        previous["started_at"] = "2026-10-14T12:00:00+00:00"
        previous["pass_policy"] = "category_only_v1"
        html = render_report(self.report, self.dataset, previous=previous,
                             previous_dataset=self.dataset)
        self.assertIn("incompatible_pass_policy", html)
        self.assertEqual(html.count("<circle "), 1)
