"""Offline tests for versioned prompts and the existing classifier contract."""

from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from app.email_classifier import DEFAULT_PROMPT, classify_email
from app.prompt_config import load_prompt, render_prompt


class PromptConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = load_prompt()
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)

    def write_config(self, config, version="v1"):
        (self.path / f"{version}.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True), encoding="utf-8"
        )

    def test_load_version_metadata(self):
        self.assertEqual(self.config["version_id"], "v1")
        self.assertIsNotNone(datetime.fromisoformat(self.config["timestamp"]).tzinfo)
        self.assertTrue(self.config["system_prompt"].strip())

    def test_required_fields(self):
        for field in ("version_id", "timestamp", "system_prompt", "few_shot_examples"):
            with self.subTest(field=field):
                config = deepcopy(self.config)
                del config[field]
                self.write_config(config)
                with self.assertRaises(ValueError):
                    load_prompt(prompts_dir=self.path)

    def test_invalid_metadata(self):
        for field, value in (
            ("version_id", "v2"), ("timestamp", "invalid"),
            ("timestamp", "2026-10-12T12:00:00"), ("timestamp", None),
            ("system_prompt", " "),
        ):
            with self.subTest(field=field, value=value):
                config = {**self.config, field: value}
                self.write_config(config)
                with self.assertRaises(ValueError):
                    load_prompt(prompts_dir=self.path)

    def test_examples_cover_all_categories_and_render(self):
        examples = self.config["few_shot_examples"]
        self.assertEqual(
            {e["expected_output"]["category"] for e in examples},
            {"billing", "technical", "account", "general"},
        )
        rendered = render_prompt(self.config)
        for example in examples:
            self.assertTrue(example["input_email"])
            self.assertIn(example["input_email"], rendered)
            self.assertIn(example["expected_output"]["summary"], rendered)
        self.assertEqual(DEFAULT_PROMPT, rendered)

    def test_invalid_examples(self):
        for examples in (
            {}, [None], [{}],
            [{"input_email": " ", "expected_output": {}}],
            [{"input_email": "Help", "expected_output": {"category": "other", "summary": "Help."}}],
            [{"input_email": "Help", "expected_output": {"category": [], "summary": "Help."}}],
            [{"input_email": "Help", "expected_output": {"category": "general", "summary": ""}}],
        ):
            with self.subTest(examples=examples):
                self.write_config({**self.config, "few_shot_examples": examples})
                with self.assertRaises(ValueError):
                    load_prompt(prompts_dir=self.path)

    def test_select_version_and_parse_unicode(self):
        config = deepcopy(self.config)
        config["version_id"] = "v2"
        config["few_shot_examples"][0]["input_email"] = "Faturamı göremiyorum."
        self.write_config(config, version="v2")
        loaded = load_prompt("v2", prompts_dir=self.path)
        self.assertEqual(loaded, config)
        self.assertIn("Faturamı göremiyorum.", render_prompt(loaded))

    def test_missing_file_and_invalid_version(self):
        with self.assertRaises(FileNotFoundError):
            load_prompt("v99", prompts_dir=self.path)
        for version in ("../v1", "v0", "", None):
            with self.subTest(version=version), self.assertRaises(ValueError):
                load_prompt(version, prompts_dir=self.path)

    def test_invalid_yaml_and_non_mapping(self):
        for content in ("[broken", "- item", "", "!!python/object:builtins.object {}"):
            with self.subTest(content=content):
                (self.path / "v1.yaml").write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_prompt(prompts_dir=self.path)

    def test_default_path_independent_of_working_directory(self):
        original = Path.cwd()
        try:
            os.chdir(self.path)
            self.assertEqual(load_prompt(), self.config)
        finally:
            os.chdir(original)


class ClassifierCompatibilityTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"GEMINI_API_KEY": "offline-test-key"})
        env.start()
        self.addCleanup(env.stop)
        client_patch = patch("app.email_classifier.genai.Client")
        self.factory = client_patch.start()
        self.addCleanup(client_patch.stop)
        self.client = self.factory.return_value.__enter__.return_value
        self.output = {"category": "technical", "summary": "The application crashes during uploads."}
        self.client.models.generate_content.return_value = SimpleNamespace(text=json.dumps(self.output))

    def test_default_call_uses_yaml_and_examples(self):
        self.assertEqual(classify_email("Upload crashes."), self.output)
        args = self.client.models.generate_content.call_args.kwargs
        self.assertEqual(args["config"]["system_instruction"], render_prompt(load_prompt("v1")))
        self.assertEqual(args["contents"], "Upload crashes.")
        self.assertEqual(args["model"], "gemini-3.6-flash")

    def test_custom_prompt_and_model_are_preserved(self):
        self.assertEqual(classify_email("Help", "Custom prompt", model="custom-model"), self.output)
        args = self.client.models.generate_content.call_args.kwargs
        self.assertEqual(args["config"]["system_instruction"], "Custom prompt")
        self.assertEqual(args["model"], "custom-model")

    def test_invalid_inputs_do_not_call_api(self):
        for kwargs in ({"email": ""}, {"email": "Hi", "prompt": ""}, {"email": "Hi", "model": ""}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                classify_email(**kwargs)
        self.factory.assert_not_called()

    def test_missing_key(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}), self.assertRaises(RuntimeError):
            classify_email("Help")
        self.factory.assert_not_called()

    def test_invalid_responses(self):
        for text in (None, "bad json", "[]", '{"category":"other","summary":"Help."}', '{"category":"billing","summary":""}'):
            self.client.models.generate_content.return_value = SimpleNamespace(text=text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                classify_email("Help")

    def test_api_error_propagates(self):
        self.client.models.generate_content.side_effect = ConnectionError("offline")
        with self.assertRaises(ConnectionError):
            classify_email("Help")


if __name__ == "__main__":
    unittest.main()
