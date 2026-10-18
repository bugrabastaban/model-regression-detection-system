"""Classifier-to-runner metadata integration with a mocked Gemini SDK only."""

import json
import os
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.contracts import ClassifierOutput
from app.email_classifier import (_classify_email_response, classify_email,
                              classify_email_raw, parse_classifier_output)
from app.evaluation_runner import run_evaluation
from app.prompt_config import load_prompt_config, render_prompt
from app.token_usage import TokenUsageResult


OUTPUT = {"category": "billing", "summary": "The customer requests a refund."}
RAW = json.dumps(OUTPUT)
METADATA = {"prompt_token_count": 12, "candidates_token_count": 4, "total_token_count": 20}
COUNTS = {"input_tokens": 12, "output_tokens": 4, "total_tokens": 20}
UNKNOWN = {"input_tokens": None, "output_tokens": None, "total_tokens": None}


def mock_sdk(test):
    env = patch.dict(os.environ, {"GEMINI_API_KEY": "offline-key"})
    env.start()
    test.addCleanup(env.stop)
    sdk = patch("app.email_classifier.genai.Client")
    factory = sdk.start()
    test.addCleanup(sdk.stop)
    return factory.return_value.__enter__.return_value


class ClassifierMetadataTests(unittest.TestCase):
    def setUp(self):
        self.client = mock_sdk(self)

    def test_internal_response_carries_text_and_usage_with_one_sdk_call(self):
        self.client.models.generate_content.return_value = SimpleNamespace(
            text=RAW, usage_metadata=SimpleNamespace(**METADATA))
        response = _classify_email_response("Refund please.", "Custom prompt", model="test-model")
        self.assertEqual(response.text, RAW)
        self.assertIsInstance(response.usage, TokenUsageResult)
        self.assertEqual(response.usage.token_usage.model_dump(), COUNTS)
        self.assertIsNone(response.usage.error)
        self.client.models.generate_content.assert_called_once_with(
            model="test-model", contents="Refund please.", config={
                "system_instruction": "Custom prompt", "response_mime_type": "application/json",
                "response_json_schema": ClassifierOutput.model_json_schema()})

    def test_public_interfaces_keep_return_types_and_make_one_sdk_call_each(self):
        for metadata in (METADATA, None, {"prompt_token_count": -1}):
            for classify, expected in ((classify_email_raw, RAW), (classify_email, OUTPUT)):
                with self.subTest(metadata=metadata, classifier=classify.__name__):
                    self.client.models.generate_content.reset_mock()
                    self.client.models.generate_content.return_value = SimpleNamespace(
                        text=RAW, usage_metadata=metadata)
                    actual = classify("Refund please.")
                    self.assertEqual(actual, expected)
                    self.assertIs(type(actual), type(expected))
                    self.client.models.generate_content.assert_called_once()

    def test_public_raw_preserves_empty_or_invalid_text_without_validation(self):
        for raw in (None, "", "bad json"):
            with self.subTest(raw=raw):
                self.client.models.generate_content.reset_mock()
                self.client.models.generate_content.return_value = SimpleNamespace(
                    text=raw, usage_metadata=METADATA)
                self.assertEqual(classify_email_raw("Help"), raw)
                self.client.models.generate_content.assert_called_once()
                self.client.models.generate_content.reset_mock()
                with self.assertRaises(ValueError):
                    classify_email("Help")
                self.client.models.generate_content.assert_called_once()

    def test_broken_response_metadata_property_does_not_hide_valid_output(self):
        class Response:
            text = RAW

            @property
            def usage_metadata(self):
                raise RuntimeError("secret provider detail")

        self.client.models.generate_content.return_value = Response()
        response = _classify_email_response("Help")
        self.assertEqual(parse_classifier_output(response.text), OUTPUT)
        self.assertEqual(response.usage.token_usage.model_dump(), UNKNOWN)
        self.assertEqual(response.usage.error, "invalid_metadata")
        self.assertNotIn("secret", response.usage.model_dump_json())
        self.client.models.generate_content.assert_called_once()


class RunnerMetadataTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = mock_sdk(self)
        self.prompt = load_prompt_config()
        self.dataset = {"dataset_version": "offline", "cases": [{"id": "a", "input": "Refund please."}]}

    async def test_sdk_metadata_variants_do_not_fail_valid_classification(self):
        variants = [({}, UNKNOWN, None),
                    ({"usage_metadata": None}, UNKNOWN, None),
                    ({"usage_metadata": {}}, UNKNOWN, None),
                    ({"usage_metadata": {"prompt_token_count": 7}},
                     {**UNKNOWN, "input_tokens": 7}, None),
                    ({"usage_metadata": SimpleNamespace(candidates_token_count=0)},
                     {**UNKNOWN, "output_tokens": 0}, None),
                    ({"usage_metadata": dict.fromkeys(METADATA, 0)},
                     dict.fromkeys(COUNTS, 0), None),
                    ({"usage_metadata": {"prompt_token_count": "secret-value"}},
                     UNKNOWN, "invalid_metadata"),
                    ({"usage_metadata": {"prompt_token_count": 12, "candidates_token_count": -1}},
                     UNKNOWN, "invalid_metadata")]
        for attributes, counts, error in variants:
            with self.subTest(attributes=attributes):
                self.client.models.generate_content.reset_mock()
                self.client.models.generate_content.return_value = SimpleNamespace(text=RAW, **attributes)
                report = await run_evaluation(self.prompt, self.dataset, batch_size=1)
                row = report["results"][0]
                self.assertEqual(row["raw_output"], RAW)
                self.assertEqual(row["output"], OUTPUT)
                self.assertIsNone(row["error"])
                self.assertEqual(row["token_usage"], counts)
                self.assertEqual(row["token_usage_error"], error)
                self.assertNotIn("secret-value", json.dumps(report, allow_nan=False))
                self.client.models.generate_content.assert_called_once()

    async def test_usage_survives_invalid_json_category_summary_and_empty_response(self):
        for raw in ("bad json", '{"category":"other","summary":"Help."}',
                    '{"category":"billing","summary":""}', None, ""):
            with self.subTest(raw=raw):
                self.client.models.generate_content.reset_mock()
                self.client.models.generate_content.return_value = SimpleNamespace(
                    text=raw, usage_metadata=METADATA)
                report = await run_evaluation(self.prompt, self.dataset, batch_size=1)
                row = report["results"][0]
                self.assertEqual(row["raw_output"], raw)
                self.assertIsNone(row["output"])
                self.assertEqual(row["error"]["type"], "ValueError")
                self.assertEqual(row["token_usage"], COUNTS)
                self.assertIsNone(row["token_usage_error"])
                self.client.models.generate_content.assert_called_once()

    async def test_metadata_and_classification_errors_are_separate(self):
        self.client.models.generate_content.return_value = SimpleNamespace(
            text="bad json", usage_metadata={"total_token_count": -1})
        row = (await run_evaluation(self.prompt, self.dataset))["results"][0]
        self.assertEqual(row["error"]["type"], "ValueError")
        self.assertEqual(row["token_usage_error"], "invalid_metadata")
        self.assertEqual(row["token_usage"], UNKNOWN)
        self.client.models.generate_content.assert_called_once()

    async def test_concurrent_out_of_order_sdk_responses_keep_their_own_usage(self):
        self.dataset["cases"] = [{"id": str(i), "input": f"Email {i}"} for i in range(2)]
        second_parsed = threading.Event()
        completions = []

        def generate(*, contents, **kwargs):
            i = int(contents[-1])
            if i == 0 and not second_parsed.wait(timeout=3):
                raise TimeoutError("Second result was not parsed")
            return SimpleNamespace(text=json.dumps({"category": "general", "summary": contents}),
                                   usage_metadata=SimpleNamespace(prompt_token_count=10 + i,
                                                                  candidates_token_count=i,
                                                                  total_token_count=20 + i))

        def parse(text):
            output = parse_classifier_output(text)
            completions.append(output["summary"])
            if output["summary"] == "Email 1":
                second_parsed.set()
            return output

        self.client.models.generate_content.side_effect = generate
        with patch("app.evaluation_runner.parse_classifier_output", side_effect=parse):
            report = await run_evaluation(self.prompt, self.dataset, model="test-model", batch_size=2)
        self.assertEqual(completions, ["Email 1", "Email 0"])
        self.assertEqual(self.client.models.generate_content.call_count, 2)
        for i, row in enumerate(report["results"]):
            self.assertEqual(row["case_id"], str(i))
            self.assertEqual(row["output"]["summary"], f"Email {i}")
            self.assertEqual(row["token_usage"], {"input_tokens": 10 + i, "output_tokens": i,
                                                 "total_tokens": 20 + i})
            self.assertIsNone(row["error"])
            self.assertIsNone(row["token_usage_error"])
        for call in self.client.models.generate_content.call_args_list:
            self.assertEqual(call.kwargs["model"], "test-model")
            self.assertEqual(call.kwargs["config"]["system_instruction"], render_prompt(self.prompt))
        json.dumps(report, allow_nan=False)
