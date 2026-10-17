"""Pure offline run comparisons; no provider calls."""

from copy import deepcopy
import json
import unittest

from comparison import PASS_POLICY, compare_runs


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.dataset = {"dataset_version": "v1", "cases": [
            {"id": key, "input": f"Email {key}", "expected_category": category,
             "ideal_summary": "A customer requests help."}
            for key, category in (("a", "billing"), ("b", "technical"), ("c", "account"))
        ]}
        self.previous = {"dataset_version": "v1", "judge_rubric_version": "rubric1",
                         "prompt_version": "v1", "results": [
            self.row("a", "billing", 5), self.row("b", "general", 5),
            self.row("c", "account", 4)]}
        self.current = deepcopy(self.previous)

    @staticmethod
    def row(key, category, score):
        return {"case_id": key, "error": None,
                "output": {"category": category, "summary": "A customer requests help."},
                "scores": {"error": None, "summary_relevance": score,
                           "judge": {"status": "scored", "error": None, "model": "judge1"}}}

    def compare(self, **kwargs):
        return compare_runs(self.current, self.previous, current_dataset=self.dataset,
                            previous_dataset=kwargs.get("previous_dataset", self.dataset))

    def test_regressions_improvements_and_pairing_ignore_order(self):
        self.current["results"] = [self.row("c", "account", 4),
                                   self.row("b", "technical", 5), self.row("a", "general", 5)]
        result = self.compare()
        self.assertEqual(result["regressions"], ["a"])
        self.assertEqual(result["improvements"], ["b"])
        self.assertEqual(result["overall_pass_rate"]["delta_percentage_points"], 0)
        self.assertEqual(result["per_category_accuracy"]["billing"]["delta_percentage_points"], -100)
        self.assertEqual(result["per_category_accuracy"]["technical"]["delta_percentage_points"], 100)

    def test_overall_delta_is_percentage_points(self):
        self.current["results"][1] = self.row("b", "technical", 5)
        result = self.compare()["overall_pass_rate"]
        self.assertAlmostEqual(result["previous_rate"], 2 / 3)
        self.assertEqual(result["current_rate"], 1)
        self.assertAlmostEqual(result["delta_percentage_points"], 100 / 3)

    def test_summary_threshold_and_category_required(self):
        self.current["results"][0]["scores"]["summary_relevance"] = 3
        result = self.compare()
        self.assertEqual(result["regressions"], ["a"])
        self.assertEqual(result["per_category_accuracy"]["billing"]["delta_percentage_points"], 0)
        self.assertEqual(result["improvements"], [])

    def test_errors_and_missing_scores_are_unevaluable(self):
        self.current["results"][0]["error"] = {"type": "TimeoutError"}
        self.current["results"][1]["scores"]["summary_relevance"] = None
        result = self.compare()
        self.assertEqual(result["unevaluable_cases"], ["a", "b"])
        self.assertEqual(result["regressions"], [])
        self.assertEqual(result["overall_pass_rate"]["compared_cases"], 1)
        self.assertEqual(result["coverage"]["category_comparable_cases"], 2)

    def test_invalid_outputs_and_judge_errors(self):
        self.current["results"][0]["output"] = {"category": "invalid", "summary": "Help."}
        self.current["results"][1]["scores"]["judge"]["error"] = {"type": "TimeoutError"}
        self.assertEqual(self.compare()["unevaluable_cases"], ["a", "b"])

    def test_changed_reference_fields_excluded_even_same_version(self):
        for field, value in (("input", "Changed email"), ("expected_category", "general"),
                             ("ideal_summary", "Changed summary.")):
            old_dataset = deepcopy(self.dataset)
            old_dataset["cases"][0][field] = value
            with self.subTest(field=field):
                result = self.compare(previous_dataset=old_dataset)
                self.assertEqual(result["changed_cases"], [{"case_id": "a", "fields": [field]}])
                self.assertEqual(result["overall_pass_rate"]["compared_cases"], 2)

    def test_notes_difficulty_and_review_edits_do_not_exclude(self):
        old_dataset = deepcopy(self.dataset)
        old_dataset["cases"][0].update(notes="Old", expected_difficulty="hard", review_status="pending")
        self.assertEqual(self.compare(previous_dataset=old_dataset)["changed_cases"], [])

    def test_added_removed_and_changed_dataset_version(self):
        old_dataset = deepcopy(self.dataset)
        self.dataset["dataset_version"] = self.current["dataset_version"] = "v2"
        self.current["results"].pop(0)
        self.current["results"].append(self.row("d", "general", 5))
        self.dataset["cases"].append({"id": "d", "input": "New", "expected_category": "general", "ideal_summary": "Help."})
        result = self.compare(previous_dataset=old_dataset)
        self.assertEqual(result["added_cases"], ["d"])
        self.assertEqual(result["removed_cases"], ["a"])
        self.assertEqual(result["overall_pass_rate"]["compared_cases"], 2)

    def test_changed_or_unknown_judge_configuration(self):
        for rubric in (None, "different"):
            self.current["judge_rubric_version"] = rubric
            result = self.compare()
            self.assertEqual(result["judge_incompatible_cases"], ["a", "b", "c"])
            self.assertIsNone(result["overall_pass_rate"]["delta_percentage_points"])
            self.assertEqual(result["coverage"]["category_comparable_cases"], 3)
        self.current["judge_rubric_version"] = "rubric1"
        self.current["results"][0]["scores"]["judge"]["model"] = "other-judge"
        self.assertEqual(self.compare()["judge_incompatible_cases"], ["a"])

    def test_empty_and_baseline_no_fabricated_delta(self):
        baseline = compare_runs(self.current, current_dataset=self.dataset)
        self.assertEqual(baseline["status"], "baseline")
        self.assertIsNone(baseline["overall_pass_rate"]["delta_percentage_points"])
        self.current["results"] = []
        result = self.compare()
        self.assertIsNone(result["overall_pass_rate"]["current_rate"])
        self.assertEqual(result["overall_pass_rate"]["compared_cases"], 0)
        self.assertIsNone(result["per_category_accuracy"]["general"]["delta_percentage_points"])

    def test_reject_ambiguous_ids_and_wrong_datasets(self):
        with self.assertRaises(ValueError):
            compare_runs(self.current, self.previous, current_dataset=self.dataset)
        self.current["results"].append(deepcopy(self.current["results"][0]))
        with self.assertRaises(ValueError):
            self.compare()
        self.current["results"].pop()
        self.current["dataset_version"] = "wrong"
        with self.assertRaises(ValueError):
            self.compare()

    def test_no_input_mutation_and_json_serialization(self):
        before = deepcopy((self.current, self.previous, self.dataset))
        json.dumps(self.compare(), allow_nan=False)
        self.assertEqual((self.current, self.previous, self.dataset), before)

    def test_legacy_and_explicit_supported_pass_policies_are_compatible(self):
        for current_explicit, previous_explicit in ((False, False), (True, False),
                                                    (False, True), (True, True)):
            with self.subTest(current=current_explicit, previous=previous_explicit):
                for report, explicit in ((self.current, current_explicit),
                                         (self.previous, previous_explicit)):
                    report.pop("pass_policy", None)
                    if explicit:
                        report["pass_policy"] = PASS_POLICY
                result = self.compare()
                self.assertEqual(result["status"], "compared")
                self.assertTrue(result["pass_policy_compatibility"]["compatible"])
                self.assertEqual(result["overall_pass_rate"]["compared_cases"], 3)

    def test_unsupported_or_invalid_pass_policy_never_produces_deltas(self):
        for side in ("current", "previous", "both"):
            for policy in ("category_only_v1", "", None, False, {}):
                with self.subTest(side=side, policy=policy):
                    current, previous = deepcopy(self.current), deepcopy(self.previous)
                    if side in ("current", "both"):
                        current["pass_policy"] = policy
                    if side in ("previous", "both"):
                        previous["pass_policy"] = policy
                    result = compare_runs(current, previous, current_dataset=self.dataset,
                                          previous_dataset=self.dataset)
                    self.assertEqual(result["status"], "incompatible_pass_policy")
                    self.assertFalse(result["pass_policy_compatibility"]["compatible"])
                    self.assertIsNone(result["overall_pass_rate"]["delta_percentage_points"])
                    self.assertEqual(result["threshold_assessment"]["severity"], "not_evaluated")
                    self.assertEqual(result["regressions"], [])
                    self.assertEqual(result["improvements"], [])
                    self.assertEqual(result["coverage"]["pass_comparable_cases"], 0)
                    self.assertTrue(all(metric["compared_cases"] == 0
                                        for metric in result["per_category_accuracy"].values()))

    def test_unsupported_baseline_policy_is_explicitly_marked(self):
        self.current["pass_policy"] = "category_only_v1"
        result = compare_runs(self.current, current_dataset=self.dataset)
        self.assertEqual(result["status"], "incompatible_pass_policy")
        self.assertIsNone(result["overall_pass_rate"]["current_rate"])
