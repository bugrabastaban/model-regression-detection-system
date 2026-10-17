"""Offline runner-to-scoring token integration and quality invariance."""

from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from comparison import compare_runs
from drift_detection import detect_slow_drift
from evaluation_runner import run_evaluation
from prompt_config import load_prompt_config
from scoring import JudgeResult, score_results, save_scored_results
from token_usage import TokenUsage


UNKNOWN = {"input_tokens": None, "output_tokens": None, "total_tokens": None}
GEMINI = {"input_tokens": 12, "output_tokens": 4, "total_tokens": 20}
GROQ = {"input_tokens": 30, "output_tokens": 5, "total_tokens": 40}


class ScoringTokenTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.dataset = {"dataset_version": "offline", "cases": [
            {"id": "a", "input": "Refund please.", "expected_category": "billing",
             "ideal_summary": "The customer requests a refund."},
            {"id": "b", "input": "Upload crashes.", "expected_category": "technical",
             "ideal_summary": "The customer reports an upload failure."}]}
        self.report = {"dataset_version": "offline", "model": "offline-model", "results": [
            {"case_id": case["id"], "output": {"category": case["expected_category"],
                                               "summary": case["ideal_summary"]},
             "error": None, "elapsed_seconds": 0.5}
            for case in self.dataset["cases"]]}

    def judge(self):
        return AsyncMock(return_value=JudgeResult(raw_output='{"score":5,"reason":"Faithful."}',
                                                 model="offline-judge", token_usage=TokenUsage(**GROQ)))

    async def test_mocked_sdk_runner_scoring_preserves_separate_provider_usage(self):
        prompt = load_prompt_config()
        references = {case["input"]: case for case in self.dataset["cases"]}

        def generate(*, contents, **kwargs):
            case = references[contents]
            return SimpleNamespace(text=json.dumps({"category": case["expected_category"],
                                                     "summary": case["ideal_summary"]}),
                                   usage_metadata=SimpleNamespace(prompt_token_count=12,
                                                                  candidates_token_count=4,
                                                                  total_token_count=20))

        with patch.dict(os.environ, {"GEMINI_API_KEY": "offline-key"}), patch(
            "email_classifier.genai.Client"
        ) as factory:
            client = factory.return_value.__enter__.return_value
            client.models.generate_content.side_effect = generate
            raw = await run_evaluation(prompt, self.dataset, batch_size=2)
        before = deepcopy(raw)
        judge = self.judge()
        scored = await score_results(raw, self.dataset, judge=judge)
        self.assertEqual(raw, before)
        self.assertEqual(client.models.generate_content.call_count, 2)
        self.assertEqual(judge.await_count, 2)
        for row in scored["results"]:
            self.assertEqual(row["scores"]["token_usage"], GEMINI)
            self.assertEqual(row["scores"]["judge"]["token_usage"], GROQ)
            self.assertIsNone(row["scores"]["token_usage_error"])

    async def test_missing_partial_zero_and_provider_total_are_preserved(self):
        for raw, expected in ((None, UNKNOWN), ({}, UNKNOWN),
                              ({"input_tokens": 7}, {**UNKNOWN, "input_tokens": 7}),
                              ({"output_tokens": 0}, {**UNKNOWN, "output_tokens": 0}),
                              (dict.fromkeys(UNKNOWN, 0), dict.fromkeys(UNKNOWN, 0)),
                              ({"input_tokens": 12, "output_tokens": 4},
                               {**UNKNOWN, "input_tokens": 12, "output_tokens": 4}),
                              (GEMINI, GEMINI)):
            with self.subTest(raw=raw):
                report = deepcopy(self.report)
                report["results"][0]["token_usage"] = raw
                result = await score_results(report, self.dataset)
                self.assertEqual(result["results"][0]["scores"]["token_usage"], expected)
                self.assertIsNone(result["results"][0]["scores"]["token_usage_error"])

    async def test_legacy_json_without_token_fields_scores_and_saves(self):
        legacy = json.loads(json.dumps(self.report))
        result = await score_results(legacy, self.dataset, judge=self.judge())
        for row in result["results"]:
            self.assertEqual(row["scores"]["category_match"], 1)
            self.assertEqual(row["scores"]["summary_relevance"], 5)
            self.assertEqual(row["scores"]["token_usage"], UNKNOWN)
            self.assertIsNone(row["scores"]["token_usage_error"])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "legacy_scored.json"
            save_scored_results(result, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), result)
        self.assertEqual(legacy, self.report)

    async def test_invalid_usage_never_aborts_scoring_or_json_serialization(self):
        for invalid in ({"input_tokens": -1}, {"output_tokens": True},
                        {"total_tokens": "secret-value"}, {"input_tokens": float("nan")},
                        {"extra": 1}, [], "bad", False, 0):
            with self.subTest(invalid=invalid):
                report = deepcopy(self.report)
                report["results"][0]["token_usage"] = invalid
                judge = self.judge()
                result = await score_results(report, self.dataset, judge=judge)
                first = result["results"][0]
                self.assertEqual(first["scores"]["token_usage"], UNKNOWN)
                self.assertEqual(first["scores"]["token_usage_error"], "invalid_metadata")
                self.assertEqual(first["scores"]["category_match"], 1)
                self.assertEqual(first["scores"]["summary_relevance"], 5)
                self.assertIsNone(first["error"])
                self.assertIsNone(first["scores"]["error"])
                self.assertEqual(judge.await_count, 2)
                serialized = json.dumps(result, allow_nan=False)
                self.assertNotIn("secret-value", serialized)
                self.assertEqual(result["results"][1]["scores"]["status"], "scored")

    async def test_runner_metadata_error_stays_separate_from_classifier_error(self):
        self.report["results"][0].update(token_usage=GEMINI, token_usage_error="invalid_metadata")
        self.report["results"][1].update(token_usage=GEMINI, output=None,
                                         error={"type": "ValueError", "message": "Invalid JSON"})
        judge = self.judge()
        result = await score_results(self.report, self.dataset, judge=judge)
        first, second = [row["scores"] for row in result["results"]]
        self.assertEqual(first["token_usage"], UNKNOWN)
        self.assertEqual(first["token_usage_error"], "invalid_metadata")
        self.assertIsNone(first["error"])
        self.assertEqual(first["summary_relevance"], 5)
        self.assertEqual(second["token_usage"], GEMINI)
        self.assertIsNone(second["token_usage_error"])
        self.assertEqual(second["status"], "classifier_error")
        self.assertEqual(judge.await_count, 1)

    async def test_tokens_do_not_change_quality_pass_regressions_thresholds_or_drift(self):
        legacy_runs, usage_runs = [], []
        for index in range(7):
            report = deepcopy(self.report)
            report.update(run_id=f"offline-{index}", started_at=f"2026-01-{index + 1:02d}T00:00:00+00:00")
            if index == 6:
                report["results"][0]["output"]["category"] = "general"
            with_usage = deepcopy(report)
            with_usage["results"][0]["token_usage"] = GEMINI if index % 2 else {"input_tokens": -1}
            with_usage["results"][1]["token_usage"] = {"input_tokens": 0}
            legacy = await score_results(report, self.dataset, judge=self.judge())
            enriched = await score_results(with_usage, self.dataset, judge=self.judge())
            for before, after in zip(legacy["results"], enriched["results"]):
                for field in ("category_match", "summary_relevance", "status", "error", "latency_seconds"):
                    self.assertEqual(before["scores"][field], after["scores"][field])
                self.assertEqual(before["scores"]["judge"]["status"], after["scores"]["judge"]["status"])
            legacy_runs.append(legacy)
            usage_runs.append(enriched)

        def compare(runs):
            return compare_runs(runs[-1], runs[-2], current_dataset=self.dataset,
                                previous_dataset=self.dataset)

        legacy_comparison, usage_comparison = compare(legacy_runs), compare(usage_runs)
        # Token reporting is additive; every quality-related field must remain
        # identical, including case states, coverage and threshold assessments.
        legacy_tokens = legacy_comparison.pop("token_usage_comparisons")
        usage_tokens = usage_comparison.pop("token_usage_comparisons")
        self.assertNotEqual(legacy_tokens, usage_tokens)
        self.assertEqual(legacy_comparison, usage_comparison)
        self.assertEqual(compare(usage_runs)["regressions"], ["a"])
        self.assertEqual(compare(usage_runs)["threshold_assessment"]["severity"], "critical")

        def drift(runs):
            return detect_slow_drift(runs[-1], self.dataset,
                                     history=[(run, self.dataset) for run in runs[:-1]])

        self.assertEqual(drift(legacy_runs), drift(usage_runs))
