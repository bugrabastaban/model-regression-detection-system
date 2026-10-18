"""Offline tests for the typed prompt and classifier output contracts."""

from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pydantic import ValidationError
import yaml

from app.contracts import ClassifierOutput
from app.email_classifier import classify_email
from app.prompt_config import PromptConfig, load_prompt, load_prompt_config, render_prompt


class PromptContractTests(unittest.TestCase):
    def test_typed_loader_preserves_yaml_and_rendering(self):
        config = load_prompt_config()
        self.assertIsInstance(config, PromptConfig)
        self.assertTrue(is_dataclass(config))
        self.assertEqual(asdict(config), load_prompt())
        self.assertEqual(render_prompt(config), render_prompt(load_prompt()))
        for example in config.few_shot_examples:
            self.assertTrue(example["input_email"])
            ClassifierOutput.model_validate(example["expected_output"])

    def test_selected_version_and_existing_validation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            data = load_prompt()
            data["version_id"] = "v2"
            file = path / "v2.yaml"
            file.write_text(yaml.safe_dump(data), encoding="utf-8")
            config = load_prompt_config("v2", prompts_dir=path)
            self.assertEqual(config.version_id, "v2")
            self.assertEqual(asdict(config), data)
            del data["timestamp"]
            file.write_text(yaml.safe_dump(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_prompt_config("v2", prompts_dir=path)


class OutputContractTests(unittest.TestCase):
    def test_valid_categories_and_summary_normalization(self):
        for category in ("billing", "technical", "account", "general"):
            with self.subTest(category=category):
                result = ClassifierOutput(category=category, summary="  The customer needs help.  ")
                self.assertEqual(result.model_dump(), {
                    "category": category, "summary": "The customer needs help."
                })

    def test_invalid_categories(self):
        for category in ("other", "Billing", "", None, [], 1):
            with self.subTest(category=category), self.assertRaises(ValidationError):
                ClassifierOutput(category=category, summary="Help is requested.")

    def test_missing_fields(self):
        for data in ({}, {"category": "billing"}, {"summary": "Help is requested."}):
            with self.subTest(data=data), self.assertRaises(ValidationError):
                ClassifierOutput.model_validate(data)

    def test_empty_and_non_string_summaries(self):
        for summary in ("", " \n\t ", None, 42, []):
            with self.subTest(summary=summary), self.assertRaises(ValidationError):
                ClassifierOutput(category="general", summary=summary)

    def test_extra_fields_rejected_and_schema_matches(self):
        with self.assertRaises(ValidationError):
            ClassifierOutput(category="general", summary="Help is requested.", extra=True)
        schema = ClassifierOutput.model_json_schema()
        self.assertEqual(set(schema["properties"]), {"category", "summary"})
        self.assertEqual(set(schema["required"]), {"category", "summary"})
        self.assertFalse(schema["additionalProperties"])

    def test_classifier_validates_output_and_keeps_dictionary_interface(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "offline-key"}), patch(
            "app.email_classifier.genai.Client"
        ) as factory:
            client = factory.return_value.__enter__.return_value
            client.models.generate_content.return_value = SimpleNamespace(
                text=json.dumps({"category": "account", "summary": "  The customer forgot a password.  "})
            )
            result = classify_email("I forgot my password.")
            self.assertIsInstance(result, dict)
            self.assertEqual(result, {"category": "account", "summary": "The customer forgot a password."})
            args = client.models.generate_content.call_args.kwargs
            self.assertEqual(args["contents"], "I forgot my password.")
            self.assertEqual(args["config"]["response_json_schema"], ClassifierOutput.model_json_schema())
            for output in (
                {"category": "other", "summary": "Help."},
                {"category": "general"},
                {"summary": "Help."},
                {"category": "general", "summary": " \t "},
                {"category": "general", "summary": "Help.", "extra": True},
            ):
                client.models.generate_content.return_value = SimpleNamespace(text=json.dumps(output))
                with self.subTest(output=output), self.assertRaises(ValueError) as error:
                    classify_email("Help")
                self.assertIsInstance(error.exception.__cause__, ValidationError)


if __name__ == "__main__":
    unittest.main()
