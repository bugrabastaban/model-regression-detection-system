"""Offline runner tests; all feature calls are mocked."""

import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import threading
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from google.genai.errors import ClientError
import httpx

from app.evaluation_runner import run_evaluation
from app.email_classifier import _ClassifierResponse
from app.prompt_config import load_prompt_config, render_prompt
from app.token_usage import TokenUsageResult


RAW = ' {"category":"general","summary":"  The customer requests help.  "} '


def response(text):
    return _ClassifierResponse(text=text, usage=TokenUsageResult())


class EvaluationRunnerTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.prompt = load_prompt_config()
        self.dataset = {
            "dataset_version": "test_v1",
            "cases": [{"id": f"case_{i}", "input": f"Help {i}"} for i in range(5)],
        }

    async def test_all_cases_raw_text_order_metadata_and_no_mutation(self):
        original = deepcopy(self.dataset)
        with patch("app.evaluation_runner._classify_email_response", return_value=response(RAW)) as feature:
            result = await run_evaluation(self.prompt, self.dataset, model="test-model", batch_size=2)
        self.assertEqual(feature.call_count, 5)
        self.assertEqual(self.dataset, original)
        self.assertEqual(result["prompt_version"], self.prompt.version_id)
        self.assertEqual(result["prompt_timestamp"], self.prompt.timestamp)
        self.assertEqual(result["dataset_version"], "test_v1")
        self.assertEqual(result["model"], "test-model")
        self.assertEqual([r["case_id"] for r in result["results"]], [c["id"] for c in original["cases"]])
        for row in result["results"]:
            self.assertEqual(row["raw_output"], RAW)
            self.assertEqual(row["output"]["summary"], "The customer requests help.")
            self.assertIsNone(row["error"])
            self.assertGreaterEqual(row["elapsed_seconds"], 0)
        for call in feature.call_args_list:
            self.assertEqual(call.kwargs, {"prompt": render_prompt(self.prompt), "model": "test-model"})
        json.dumps(result)

    async def test_concurrent_calls_are_bounded(self):
        self.dataset["cases"] = self.dataset["cases"][:4]
        loop = asyncio.get_running_loop()
        reached = asyncio.Event()
        release = threading.Event()
        lock = threading.Lock()
        active = peak = started = 0

        def feature(*args, **kwargs):
            nonlocal active, peak, started
            with lock:
                active += 1
                started += 1
                peak = max(peak, active)
                if active == 2:
                    loop.call_soon_threadsafe(reached.set)
            try:
                if not release.wait(timeout=5):
                    raise TimeoutError("Test release timed out")
                return response(RAW)
            finally:
                with lock:
                    active -= 1

        with patch("app.evaluation_runner._classify_email_response", side_effect=feature):
            task = asyncio.create_task(run_evaluation(self.prompt, self.dataset, batch_size=2))
            try:
                await asyncio.wait_for(reached.wait(), timeout=3)
                self.assertEqual(started, 2)
                self.assertEqual(peak, 2)
            finally:
                release.set()
                result = await task
        self.assertEqual(started, 4)
        self.assertLessEqual(peak, 2)
        self.assertTrue(all(r["error"] is None for r in result["results"]))

    async def test_failures_preserve_raw_outputs_and_continue(self):
        responses = [ConnectionError("offline failure"), response("bad json"), response(None),
                     response('{"category":"other","summary":"Help."}'), response(RAW)]
        with patch("app.evaluation_runner._classify_email_response", side_effect=responses) as feature:
            report = await run_evaluation(self.prompt, self.dataset, batch_size=1)
        rows = report["results"]
        self.assertEqual(feature.call_count, 5)
        self.assertEqual(rows[0]["error"]["type"], "ConnectionError")
        self.assertEqual(rows[1]["raw_output"], "bad json")
        self.assertEqual(rows[3]["raw_output"], responses[3].text)
        self.assertTrue(all(r["error"] and r["output"] is None for r in rows[:4]))
        self.assertIsNone(rows[4]["error"])

    async def test_preflight_errors_never_call_feature(self):
        with patch("app.evaluation_runner._classify_email_response") as feature:
            for batch_size in (0, -1, True, 1.5):
                with self.subTest(batch_size=batch_size), self.assertRaises(ValueError):
                    await run_evaluation(self.prompt, self.dataset, batch_size=batch_size)
            for dataset in ({}, {"dataset_version": "v1", "cases": [None]},
                            {"dataset_version": "v1", "cases": [{"id": "x", "input": " "}]},
                            {"dataset_version": "v1", "cases": [self.dataset["cases"][0]] * 2}):
                with self.subTest(dataset=dataset), self.assertRaises(ValueError):
                    await run_evaluation(self.prompt, dataset)
            feature.assert_not_called()

    async def test_empty_dataset(self):
        with patch("app.evaluation_runner._classify_email_response") as feature:
            result = await run_evaluation(self.prompt, {"dataset_version": "empty", "cases": []})
        self.assertEqual(result["results"], [])
        feature.assert_not_called()

    async def test_real_dataset_offline(self):
        path = Path(__file__).resolve().parents[1] / "datasets" / "golden_v1.json"
        dataset = json.loads(path.read_text(encoding="utf-8-sig"))
        with patch("app.evaluation_runner._classify_email_response", return_value=response(RAW)) as feature:
            result = await run_evaluation(self.prompt, dataset)
        self.assertEqual(len(result["results"]), len(dataset["cases"]))
        self.assertEqual(feature.call_count, len(dataset["cases"]))

    async def test_out_of_order_completion_preserves_case_output_pairing(self):
        self.dataset["cases"] = self.dataset["cases"][:2]
        second_parsed = threading.Event()
        completion_order = []
        from app.evaluation_runner import parse_classifier_output

        def feature(email, **kwargs):
            if email == "Help 0" and not second_parsed.wait(timeout=3):
                raise TimeoutError("Second result was not parsed")
            return response(json.dumps({"category": "general", "summary": email}))

        def parse(text):
            result = parse_classifier_output(text)
            completion_order.append(result["summary"])
            if result["summary"] == "Help 1":
                second_parsed.set()
            return result

        with patch("app.evaluation_runner._classify_email_response", side_effect=feature), patch(
            "app.evaluation_runner.parse_classifier_output", side_effect=parse
        ):
            report = await run_evaluation(self.prompt, self.dataset, batch_size=2)
        self.assertEqual(completion_order, ["Help 1", "Help 0"])
        self.assertEqual([(r["case_id"], r["output"]["summary"]) for r in report["results"]],
                         [("case_0", "Help 0"), ("case_1", "Help 1")])

    async def test_real_classifier_path_with_mocked_sdk_errors_and_timeouts(self):
        responses = {
            "Help 0": ClientError(429, {"error": {"status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded"}}),
            "Help 1": httpx.ReadTimeout("Provider read timed out"),
            "Help 2": TimeoutError("Deadline exceeded"),
            "Help 3": '{"category":"other","summary":"Help."}',
            "Help 4": RAW,
        }

        def generate(**kwargs):
            response = responses[kwargs["contents"]]
            if isinstance(response, Exception):
                raise response
            return SimpleNamespace(text=response)

        with patch.dict(os.environ, {"GEMINI_API_KEY": "offline-test-key"}), patch(
            "app.email_classifier.genai.Client"
        ) as factory:
            client = factory.return_value.__enter__.return_value
            client.models.generate_content.side_effect = generate
            report = await run_evaluation(self.prompt, self.dataset, model="test-model", batch_size=3)
        self.assertEqual(client.models.generate_content.call_count, 5)
        rows = report["results"]
        self.assertEqual([r["error"]["type"] for r in rows[:4]],
                         ["ClientError", "ReadTimeout", "TimeoutError", "ValueError"])
        self.assertIn("429", rows[0]["error"]["message"])
        self.assertIn("Quota exceeded", rows[0]["error"]["message"])
        self.assertIn("Provider read timed out", rows[1]["error"]["message"])
        self.assertIn("ValidationError", rows[3]["error"]["traceback"])
        self.assertIn("category", rows[3]["error"]["traceback"])
        for row in rows:
            self.assertEqual(set(row), {"case_id", "raw_output", "output", "error", "elapsed_seconds",
                                        "token_usage", "token_usage_error"})
            self.assertGreaterEqual(row["elapsed_seconds"], 0)
            self.assertEqual(row["token_usage"], {"input_tokens": None, "output_tokens": None,
                                                 "total_tokens": None})
            self.assertIsNone(row["token_usage_error"])
        for row in rows[:3]:
            self.assertIsNone(row["raw_output"])
            self.assertIsNone(row["output"])
        self.assertEqual(rows[3]["raw_output"], responses["Help 3"])
        self.assertIsNone(rows[3]["output"])
        self.assertEqual(rows[4]["raw_output"], RAW)
        self.assertIsNone(rows[4]["error"])
        for call in client.models.generate_content.call_args_list:
            self.assertEqual(call.kwargs["model"], "test-model")
            self.assertEqual(call.kwargs["config"]["system_instruction"], render_prompt(self.prompt))
        json.dumps(report)

    async def test_json_error_preserves_underlying_location(self):
        with patch("app.evaluation_runner._classify_email_response", return_value=response("bad json")):
            report = await run_evaluation(self.prompt, self.dataset)
        for row in report["results"]:
            self.assertIn("JSONDecodeError", row["error"]["traceback"])
            self.assertIn("line 1 column 1", row["error"]["traceback"])

    async def test_cancellation_is_not_recorded_as_case_failure(self):
        with patch("app.evaluation_runner._classify_email_response", side_effect=asyncio.CancelledError):
            with self.assertRaises(asyncio.CancelledError):
                await run_evaluation(self.prompt, self.dataset, batch_size=1)

    async def test_manual_api_script_is_safe_to_import(self):
        path = Path(__file__).with_name("test_classifier.py")
        with patch("app.email_classifier.classify_email") as classify:
            runpy.run_path(str(path), run_name="offline_import_check")
        classify.assert_not_called()
