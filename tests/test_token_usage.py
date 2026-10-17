"""Offline shared token contracts and backward-compatible scoring exports."""

from copy import deepcopy
import json
from types import SimpleNamespace
import unittest

from pydantic import ValidationError

import scoring
from token_usage import (TokenUsage, TokenUsageResult, extract_gemini_token_usage,
                         gemini_token_usage)


UNKNOWN = {"input_tokens": None, "output_tokens": None, "total_tokens": None}


class TokenUsageTests(unittest.TestCase):
    def test_scoring_exports_same_model_and_function(self):
        self.assertIs(scoring.TokenUsage, TokenUsage)
        self.assertIs(scoring.gemini_token_usage, gemini_token_usage)
        usage = TokenUsage(input_tokens=12)
        result = scoring.JudgeResult(raw_output="{}", model="offline", token_usage=usage)
        self.assertIs(result.token_usage, usage)

    def test_mapping_accepts_dictionary_and_sdk_object_without_mutation(self):
        metadata = {"prompt_token_count": 12, "candidates_token_count": 4,
                    "total_token_count": 20, "thoughts_token_count": 4}
        original = deepcopy(metadata)
        expected = {"input_tokens": 12, "output_tokens": 4, "total_tokens": 20}
        for value in (metadata, SimpleNamespace(**metadata)):
            self.assertEqual(gemini_token_usage(value).model_dump(), expected)
            extracted = extract_gemini_token_usage(value)
            self.assertEqual(extracted.token_usage.model_dump(), expected)
            self.assertIsNone(extracted.error)
        self.assertEqual(metadata, original)

    def test_missing_metadata_and_fields_are_null_not_errors(self):
        for metadata in (None, {}, SimpleNamespace()):
            with self.subTest(metadata=metadata):
                self.assertEqual(gemini_token_usage(metadata).model_dump(), UNKNOWN)
                result = extract_gemini_token_usage(metadata)
                self.assertEqual(result.token_usage.model_dump(), UNKNOWN)
                self.assertIsNone(result.error)

    def test_partial_metadata_and_zero_counts_are_preserved(self):
        for field, target in (("prompt_token_count", "input_tokens"),
                              ("candidates_token_count", "output_tokens"),
                              ("total_token_count", "total_tokens")):
            for count in (0, 7):
                with self.subTest(field=field, count=count):
                    expected = {**UNKNOWN, target: count}
                    self.assertEqual(gemini_token_usage({field: count}).model_dump(), expected)
                    result = extract_gemini_token_usage(SimpleNamespace(**{field: count}))
                    self.assertEqual(result.token_usage.model_dump(), expected)
                    self.assertIsNone(result.error)

    def test_total_is_never_computed_or_corrected(self):
        for total in (None, 0, 20):
            metadata = {"prompt_token_count": 12, "candidates_token_count": 4}
            if total is not None:
                metadata["total_token_count"] = total
            self.assertEqual(gemini_token_usage(metadata).total_tokens, total)
            self.assertEqual(extract_gemini_token_usage(metadata).token_usage.total_tokens, total)

    def test_strict_contract_and_legacy_parser_still_reject_invalid_counts(self):
        for field, source in (("input_tokens", "prompt_token_count"),
                              ("output_tokens", "candidates_token_count"),
                              ("total_tokens", "total_token_count")):
            for invalid in (-1, True, "3", 2.5, float("nan"), float("inf")):
                with self.subTest(field=field, invalid=invalid):
                    with self.assertRaises(ValidationError):
                        TokenUsage(**{field: invalid})
                    with self.assertRaises(ValidationError):
                        scoring.gemini_token_usage({source: invalid})
        with self.assertRaises(ValidationError):
            TokenUsage(estimated_tokens=1)

    def test_safe_extraction_reports_invalid_metadata_without_provider_values(self):
        for invalid in (-1, True, "secret-value", 2.5):
            result = extract_gemini_token_usage({"prompt_token_count": 12,
                                                "candidates_token_count": invalid})
            self.assertEqual(result.token_usage.model_dump(), UNKNOWN)
            self.assertEqual(result.error, "invalid_metadata")
            self.assertNotIn("secret-value", result.model_dump_json())

    def test_metadata_access_errors_are_safe_but_process_control_propagates(self):
        class BrokenMetadata:
            @property
            def prompt_token_count(self):
                raise RuntimeError("secret provider data")

        result = extract_gemini_token_usage(BrokenMetadata())
        self.assertEqual(result.error, "invalid_metadata")
        self.assertEqual(result.token_usage.model_dump(), UNKNOWN)
        self.assertNotIn("secret", result.model_dump_json())

        class InterruptedMetadata:
            @property
            def prompt_token_count(self):
                raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            extract_gemini_token_usage(InterruptedMetadata())

    def test_result_round_trip_and_defaults_are_independent(self):
        result = extract_gemini_token_usage({"prompt_token_count": 0})
        serialized = json.dumps(result.model_dump(), allow_nan=False)
        self.assertEqual(TokenUsageResult.model_validate_json(serialized), result)
        first, second = TokenUsageResult(), TokenUsageResult()
        first.token_usage.input_tokens = 1
        self.assertEqual(second.token_usage.model_dump(), UNKNOWN)
        with self.assertRaises(ValidationError):
            TokenUsageResult(error="secret provider exception")
