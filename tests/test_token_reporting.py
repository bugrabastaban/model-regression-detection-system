"""Offline token diffs and HTML presentation, independent of quality scoring."""

from copy import deepcopy
from html.parser import HTMLParser
import json
import unittest

from comparison import compare_runs
from html_report import render_report


UNKNOWN = {"input_tokens": None, "output_tokens": None, "total_tokens": None}


class TokenReportingTests(unittest.TestCase):
    def setUp(self):
        self.dataset = {"dataset_version": "offline", "cases": [
            {"id": key, "input": f"Email {key}", "expected_category": category,
             "ideal_summary": "The customer requests help."}
            for key, category in (("a", "billing"), ("b", "technical"))]}
        self.previous = {"run_id": "old", "started_at": "2026-01-01T00:00:00+00:00",
                         "model": "offline-model", "prompt_version": "v1",
                         "dataset_version": "offline", "judge_rubric_version": "v1", "results": [
            {"case_id": case["id"], "output": {"category": case["expected_category"],
                                               "summary": case["ideal_summary"]},
             "error": None, "elapsed_seconds": 0.5,
             "scores": {"category_match": 1, "summary_relevance": 5, "error": None,
                        "judge": {"status": "scored", "model": "offline-judge", "error": None}}}
            for case in self.dataset["cases"]]}
        self.current = deepcopy(self.previous)
        self.current.update(run_id="new", started_at="2026-01-02T00:00:00+00:00")

    def compare(self, **kwargs):
        return compare_runs(self.current, self.previous, current_dataset=self.dataset,
                            previous_dataset=kwargs.get("previous_dataset", self.dataset))

    def render(self, **kwargs):
        return render_report(self.current, self.dataset, previous=self.previous,
                             previous_dataset=kwargs.get("previous_dataset", self.dataset))

    @staticmethod
    def usage(row, gemini, groq=None):
        row["scores"]["token_usage"] = gemini
        if groq is not None:
            row["scores"]["judge"]["token_usage"] = groq

    def test_previous_current_deltas_are_separate_per_provider_and_counter(self):
        old_gemini = {"input_tokens": 12, "output_tokens": 4, "total_tokens": 20}
        new_gemini = {"input_tokens": 15, "output_tokens": 2, "total_tokens": 22}
        old_groq = {"input_tokens": 30, "output_tokens": 3, "total_tokens": 35}
        new_groq = {"input_tokens": 0, "output_tokens": 4, "total_tokens": 40}
        self.usage(self.previous["results"][0], old_gemini, old_groq)
        self.usage(self.current["results"][0], new_gemini, new_groq)
        before = deepcopy((self.previous, self.current, self.dataset))
        diff = self.compare()
        entry = diff["token_usage_comparisons"][0]
        self.assertEqual(entry["case_id"], "a")
        self.assertEqual(entry["gemini"], {"previous": old_gemini, "current": new_gemini,
                                         "delta_tokens": {"input_tokens": 3, "output_tokens": -2,
                                                          "total_tokens": 2}})
        self.assertEqual(entry["groq"], {"previous": old_groq, "current": new_groq,
                                       "delta_tokens": {"input_tokens": -30, "output_tokens": 1,
                                                        "total_tokens": 5}})
        json.dumps(diff, allow_nan=False)
        self.assertEqual((self.previous, self.current, self.dataset), before)

    def test_missing_counters_and_partial_metadata_have_null_deltas(self):
        self.usage(self.previous["results"][0], {"input_tokens": 0, "total_tokens": 5})
        self.usage(self.current["results"][0], {"input_tokens": 0, "output_tokens": 0})
        entry = self.compare()["token_usage_comparisons"][0]
        self.assertEqual(entry["gemini"]["previous"], {**UNKNOWN, "input_tokens": 0, "total_tokens": 5})
        self.assertEqual(entry["gemini"]["current"], {**UNKNOWN, "input_tokens": 0, "output_tokens": 0})
        self.assertEqual(entry["gemini"]["delta_tokens"], {**UNKNOWN, "input_tokens": 0})
        self.assertEqual(entry["groq"]["delta_tokens"], UNKNOWN)

    def test_no_baseline_has_no_fabricated_token_comparisons(self):
        self.usage(self.current["results"][0], {"total_tokens": 9})
        diff = compare_runs(self.current, current_dataset=self.dataset)
        self.assertEqual(diff["status"], "baseline")
        self.assertEqual(diff["token_usage_comparisons"], [])
        html = render_report(self.current, self.dataset)
        self.assertIn("uygun baseline kaydı yok", html)
        self.assertIn("Input: bilinmiyor<br>Output: bilinmiyor<br>Total: 9", html)

    def test_legacy_json_baseline_remains_quality_comparable(self):
        self.previous = json.loads(json.dumps(self.previous))
        self.usage(self.current["results"][0], {"total_tokens": 9})
        diff = self.compare()
        self.assertEqual(diff["overall_pass_rate"]["compared_cases"], 2)
        self.assertEqual(diff["overall_pass_rate"]["delta_percentage_points"], 0)
        for entry in diff["token_usage_comparisons"]:
            self.assertEqual(entry["gemini"]["previous"], UNKNOWN)
            self.assertEqual(entry["gemini"]["delta_tokens"], UNKNOWN)
        HTMLParser().feed(self.render())

    def test_token_matching_uses_ids_and_excludes_changed_references(self):
        self.usage(self.previous["results"][0], {"total_tokens": 10})
        self.usage(self.previous["results"][1], {"total_tokens": 20})
        self.usage(self.current["results"][0], {"total_tokens": 12})
        self.usage(self.current["results"][1], {"total_tokens": 25})
        self.current["results"].reverse()
        diff = self.compare()["token_usage_comparisons"]
        self.assertEqual([(e["case_id"], e["gemini"]["delta_tokens"]["total_tokens"]) for e in diff],
                         [("b", 5), ("a", 2)])
        old_dataset = deepcopy(self.dataset)
        old_dataset["cases"][0]["input"] = "Changed reference"
        diff = self.compare(previous_dataset=old_dataset)
        self.assertEqual([entry["case_id"] for entry in diff["token_usage_comparisons"]], ["b"])

    def test_usage_survives_quality_errors_and_incompatible_judge(self):
        self.usage(self.previous["results"][0], {"total_tokens": 10}, {"total_tokens": 4})
        self.usage(self.current["results"][0], {"total_tokens": 12}, {"total_tokens": 8})
        self.current["results"][0]["error"] = {"type": "ValueError"}
        self.current["results"][0]["scores"]["judge"]["error"] = {"type": "ValueError"}
        self.current["results"][1]["scores"]["judge"]["model"] = "other-judge"
        diff = self.compare()
        self.assertEqual(len(diff["token_usage_comparisons"]), 2)
        self.assertEqual(diff["token_usage_comparisons"][0]["groq"]["delta_tokens"]["total_tokens"], 4)
        self.assertEqual(diff["regressions"], [])
        self.assertEqual(diff["overall_pass_rate"]["compared_cases"], 0)

    def test_legacy_runner_fallback_does_not_override_scored_unknowns(self):
        row = self.current["results"][0]
        row["token_usage"] = {"total_tokens": 12}
        self.assertEqual(self.compare()["token_usage_comparisons"][0]["gemini"]["current"]["total_tokens"], 12)
        row["scores"]["token_usage"] = None
        self.assertEqual(self.compare()["token_usage_comparisons"][0]["gemini"]["current"], UNKNOWN)
        row["scores"]["token_usage"] = {"total_tokens": 12}
        row["scores"]["token_usage_error"] = "invalid_metadata"
        self.assertEqual(self.compare()["token_usage_comparisons"][0]["gemini"]["current"], UNKNOWN)

    def test_invalid_counts_do_not_break_quality_or_leak_into_html(self):
        for invalid in ({"total_tokens": -1}, {"input_tokens": True},
                        {"total_tokens": float("nan")}, {"output_tokens": '<script>secret</script>'}, []):
            with self.subTest(invalid=invalid):
                self.usage(self.current["results"][0], invalid, invalid)
                diff = self.compare()
                entry = diff["token_usage_comparisons"][0]
                self.assertEqual(entry["gemini"]["current"], UNKNOWN)
                self.assertEqual(entry["groq"]["current"], UNKNOWN)
                self.assertEqual(diff["overall_pass_rate"]["current_rate"], 1)
                json.dumps(diff, allow_nan=False)
                self.assertNotIn("<script>", self.render())

    def test_html_shows_partial_and_zero_counts_with_actual_coverage(self):
        self.usage(self.current["results"][0], {"input_tokens": 0, "total_tokens": 17},
                   {"input_tokens": 30, "output_tokens": 4})
        html = self.render()
        self.assertIn("<th>Gemini tokenları</th><th>Groq judge tokenları</th>", html)
        self.assertIn("<td>Input: 0<br>Output: bilinmiyor<br>Total: 17</td>", html)
        self.assertIn("<td>Input: 30<br>Output: 4<br>Total: bilinmiyor</td>", html)
        self.assertIn("Gemini: input 1/2, output 0/2, total 1/2", html)
        self.assertIn("Groq judge: input 1/2, output 1/2, total 0/2", html)
        self.assertNotIn("Gemini tokenları mevcut kaynakta bulunmuyor", html)
        self.assertIn("<td>a</td><td>Gemini</td><td>Input</td><td>bilinmiyor</td><td>0</td><td>bilinmiyor</td>", html)

    def test_html_token_diffs_include_negative_positive_and_zero_values(self):
        self.usage(self.previous["results"][0], {"input_tokens": 5, "output_tokens": 2, "total_tokens": 10})
        self.usage(self.current["results"][0], {"input_tokens": 3, "output_tokens": 2, "total_tokens": 15})
        html = self.render()
        for counter, before, after, delta in (("Input", 5, 3, "-2"), ("Output", 2, 2, "+0"),
                                             ("Total", 10, 15, "+5")):
            self.assertIn(f'<td>a</td><td>Gemini</td><td>{counter}</td><td>{before}</td>'
                          f'<td>{after}</td><td>{delta}</td>', html)

    def test_old_html_without_any_usage_remains_available_and_honest(self):
        html = self.render()
        self.assertIn("Input: bilinmiyor<br>Output: bilinmiyor<br>Total: bilinmiyor", html)
        self.assertIn("Gemini: input 0/2, output 0/2, total 0/2", html)
        self.assertIn("Groq judge: input 0/2, output 0/2, total 0/2", html)
        self.assertIn("100.0%", html)
        self.assertIn("bilinmiyor", html)

    def test_all_quality_fields_and_html_scorecard_remain_unchanged(self):
        self.current["results"][0]["output"]["category"] = "general"
        before = self.compare()
        old_html = self.render()
        self.usage(self.previous["results"][0], {"total_tokens": 20}, {"total_tokens": 50})
        self.usage(self.current["results"][0], {"total_tokens": 0}, {"total_tokens": 10})
        after = self.compare()
        old_tokens = before.pop("token_usage_comparisons")
        new_tokens = after.pop("token_usage_comparisons")
        self.assertNotEqual(old_tokens, new_tokens)
        self.assertEqual(before, after)
        self.assertEqual(after["regressions"], ["a"])
        self.assertEqual(after["threshold_assessment"]["severity"], "critical")

        def scorecard(html):
            return html.split('<div class="cards">', 1)[1].split('</div><section>', 1)[0]

        self.assertEqual(scorecard(old_html), scorecard(self.render()))
