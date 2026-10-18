"""Offline regression checks for package paths and command entry points."""

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from app.groq_judge import GroqJudge
from app.prompt_config import load_prompt_config
from app.slack_alerts import SlackAlertError, send_slack_alert


ROOT = Path(__file__).resolve().parents[1]


class PackageLayoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_paths_are_independent_of_working_directory(self):
        original = Path.cwd()
        with TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.assertEqual(load_prompt_config("v1").version_id, "v1")
                self.assertEqual(load_prompt_config("v2").version_id, "v2")
                with patch.dict(os.environ, {}, clear=True), patch(
                    "app.groq_judge.dotenv_values", return_value={"GROQ_API_KEY": "offline"}
                ) as values:
                    GroqJudge()
                    values.assert_called_once_with(ROOT / ".env")
                with patch.dict(os.environ, {}, clear=True), patch(
                    "app.slack_alerts.dotenv_values", return_value={}
                ) as values:
                    with self.assertRaises(SlackAlertError):
                        await send_slack_alert({})
                    values.assert_called_once_with(ROOT / ".env")
            finally:
                os.chdir(original)

    async def test_cli_help_is_available_without_provider_calls(self):
        commands = [
            ["ci_pipeline.py"],
            ["-m", "ci.pipeline"],
            ["-m", "app.html_report"],
            ["-m", "app.slack_alerts"],
        ]
        for command in commands:
            with self.subTest(command=command):
                result = subprocess.run(
                    [sys.executable, "-B", *command, "--help"],
                    cwd=ROOT, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)
