"""Offline scoring contract tests with synthetic reports and fake judges."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock

from pydantic import ValidationError

from app.scoring import JudgeResult, JudgeScore, TokenUsage, gemini_token_usage, score_results, save_scored_results


class ScoringTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.dataset = {"dataset_version": "v1", "cases": [
            {"id": "a", "input": "Refund please.", "expected_category": "billing", "ideal_summary": "The customer requests a refund."},
            {"id": "b", "input": "App crashes.", "expected_category": "technical", "ideal_summary": "The customer reports a crash."},
        ]}
        self.report = {"dataset_version": "v1", "results": [
            {"case_id": "b", "output": {"category": "general", "summary": "The customer reports a crash."}, "error": None, "elapsed_seconds": 0.5},
            {"case_id": "a", "output": {"category": "billing", "summary": "The customer requests a refund."}, "error": None, "elapsed_seconds": 1.5},
        ]}

    async def test_binary_match_join_order_and_no_mutation(self):
        before = deepcopy(self.report)
        result = await score_results(self.report, self.dataset)
        self.assertEqual([r["scores"]["category_match"] for r in result["results"]], [0, 1])
        self.assertEqual(self.report, before)
        for row in result["results"]:
            self.assertEqual(row["scores"]["latency_seconds"], row["elapsed_seconds"])
            self.assertIsNone(row["scores"]["summary_relevance"])
            self.assertIsNone(row["scores"]["token_usage"]["total_tokens"])
            self.assertEqual(row["scores"]["judge"]["status"], "not_configured")

    async def test_fake_judge_scores_and_separate_usage(self):
        self.report["results"][0]["token_usage"] = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 18}
        judge = AsyncMock(return_value=JudgeResult(raw_output='{"score":5,"reason":"Faithful."}', model="fake-judge",
                                                  token_usage=TokenUsage(input_tokens=20, output_tokens=3, total_tokens=23)))
        result = await score_results(self.report, self.dataset, judge=judge)
        self.assertEqual(judge.await_count, 2)
        score = result["results"][0]["scores"]
        self.assertEqual(score["summary_relevance"], 5)
        self.assertEqual(score["token_usage"]["total_tokens"], 18)
        self.assertEqual(score["judge"]["token_usage"]["total_tokens"], 23)
        self.assertGreaterEqual(score["judge"]["latency_seconds"], 0)
        payload = json.loads(judge.call_args_list[0].args[1])
        self.assertEqual(payload["email"], "App crashes.")
        self.assertEqual(payload["reference_summary"], self.dataset["cases"][1]["ideal_summary"])

    async def test_classifier_failure_is_not_zero_and_skips_judge(self):
        self.report["results"][0].update(output=None, error={"type": "TimeoutError"})
        self.report["results"][1]["output"] = {"category": "invalid", "summary": "Help."}
        judge = AsyncMock()
        result = await score_results(self.report, self.dataset, judge=judge)
        judge.assert_not_awaited()
        for row in result["results"]:
            self.assertIsNone(row["scores"]["category_match"])
            self.assertEqual(row["scores"]["status"], "classifier_error")

    async def test_judge_failure_does_not_erase_category_or_abort(self):
        judge = AsyncMock(side_effect=[TimeoutError("offline timeout"), JudgeResult(raw_output='{"score":4,"reason":"Minor omission."}', model="fake")])
        result = await score_results(self.report, self.dataset, judge=judge)
        self.assertEqual(judge.await_count, 2)
        first, second = [r["scores"] for r in result["results"]]
        self.assertEqual(first["category_match"], 0)
        self.assertIsNone(first["summary_relevance"])
        self.assertEqual(first["judge"]["error"]["type"], "TimeoutError")
        self.assertEqual(second["summary_relevance"], 4)

    async def test_invalid_judge_output_preserved(self):
        for raw in ('bad json', '{"score":6,"reason":"Bad."}', '{"score":3.5,"reason":"Bad."}'):
            with self.subTest(raw=raw):
                judge = AsyncMock(return_value=JudgeResult(raw_output=raw, model="fake"))
                result = await score_results(self.report, self.dataset, judge=judge)
                score = result["results"][0]["scores"]
                self.assertIsNone(score["summary_relevance"])
                self.assertEqual(score["judge"]["raw_output"], raw)
                self.assertEqual(score["judge"]["status"], "error")

    async def test_score_contract(self):
        for score in range(1, 6):
            self.assertEqual(JudgeScore(score=score, reason="Valid.").score, score)
        for score in (0, 6, True, "3", 2.5):
            with self.subTest(score=score), self.assertRaises(ValidationError):
                JudgeScore(score=score, reason="Bad.")
        with self.assertRaises(ValidationError):
            JudgeScore(score=3, reason=" ")

    async def test_token_metadata_and_missing_values(self):
        metadata = {"prompt_token_count": 12, "candidates_token_count": 4, "total_token_count": 20}
        self.assertEqual(gemini_token_usage(metadata), gemini_token_usage(SimpleNamespace(**metadata)))
        self.assertEqual(gemini_token_usage(metadata).total_tokens, 20)
        self.assertIsNone(gemini_token_usage(None).total_tokens)
        self.assertIsNone(gemini_token_usage({"prompt_token_count": 0}).output_tokens)
        with self.assertRaises(ValidationError):
            TokenUsage(input_tokens=-1)

    async def test_preflight_before_judge(self):
        judge = AsyncMock()
        variants = []
        for value in (-1, float("nan"), float("inf"), True):
            report = deepcopy(self.report)
            report["results"][1]["elapsed_seconds"] = value
            variants.append(report)
        report = deepcopy(self.report)
        report["results"][1]["case_id"] = "unknown"
        variants.append(report)
        report = deepcopy(self.report)
        report["dataset_version"] = "wrong"
        variants.append(report)
        report = deepcopy(self.report)
        report["results"].append(report["results"][0])
        variants.append(report)
        for report in variants:
            with self.assertRaises(ValueError):
                await score_results(report, self.dataset, judge=judge)
        judge.assert_not_awaited()

    async def test_json_storage_and_no_overwrite(self):
        result = await score_results(self.report, self.dataset)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "scored.json"
            save_scored_results(result, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), result)
            with self.assertRaises(FileExistsError):
                save_scored_results(result, path)
