"""Slack payload and delivery tests using MockTransport only."""

from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

import httpx

from slack_alerts import SlackAlertError, build_slack_message, send_slack_alert
from thresholds import ThresholdConfig


class SlackAlertTests(IsolatedAsyncioTestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.report = json.loads((root / "test_results/combined_subset_live_20261015.json").read_text(encoding="utf-8"))
        self.dataset = json.loads((root / "datasets/golden_v1.json").read_text(encoding="utf-8-sig"))
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.env = Path(directory.name) / ".env"
        self.env.write_text("SLACK_WEBHOOK_URL=https://hooks.slack.com/services/TEST/TEST/secret\n", encoding="utf-8")
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def build(self, current=None, previous=None, **kwargs):
        return build_slack_message(current or self.report, self.dataset, previous=previous,
                                   previous_dataset=self.dataset if previous is not None else None, **kwargs)

    async def test_baseline_is_not_claimed_as_pass_and_missing_link_is_explicit(self):
        payload = self.build()
        self.assertTrue(payload["text"].startswith("NOT EVALUATED"))
        self.assertIn("Baseline yok", payload["text"])
        self.assertIn("3/92", payload["text"])
        self.assertIn("100.0%", payload["text"])
        self.assertIn("4.67/5", payload["text"])
        self.assertIn("bağlantısı yapılandırılmadı", payload["text"])
        self.assertNotIn("0 gerileme", payload["text"])

    async def test_pass_warn_fail_and_headline_numbers(self):
        unchanged = self.build(previous=self.report)
        self.assertTrue(unchanged["text"].startswith("PASS"))
        current = deepcopy(self.report)
        current["results"][0]["output"]["category"] = "general"
        failed = self.build(current, self.report)
        self.assertTrue(failed["text"].startswith("FAIL"))
        self.assertIn("1 gerileme", failed["text"])
        self.assertIn("100.0% → 66.7%", failed["text"])
        self.assertIn("billing doğruluğu: 100.0% → 50.0%", failed["text"])
        warning = self.build(current, self.report, thresholds=ThresholdConfig(1, 60))
        self.assertTrue(warning["text"].startswith("WARN"))

    async def test_coverage_gap_warns_even_if_remaining_cases_pass(self):
        current = deepcopy(self.report)
        current["results"][0]["error"] = {"type": "TimeoutError"}
        self.assertTrue(self.build(current, self.report)["text"].startswith("WARN"))

    async def test_report_link_and_no_mutation(self):
        before = deepcopy(self.report)
        payload = self.build(report_url="https://reports.example.com/run.html")
        self.assertEqual(payload["blocks"][-1]["elements"][0]["url"], "https://reports.example.com/run.html")
        self.assertEqual(self.report, before)
        self.assertFalse(payload["unfurl_links"])

    async def test_bad_or_local_report_links_rejected(self):
        for url in ("file:///C:/report.html", "http://example.com", "https://localhost/report",
                    "https://127.0.0.1/report", "https://192.168.1.2/report",
                    "javascript:alert(1)", "https://user:secret@example.com/report"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.build(report_url=url)

    async def test_untrusted_metadata_is_plain_text(self):
        self.report["model"] = "<!channel> & <malicious>"
        payload = self.build()
        context = payload["blocks"][2]["elements"][0]
        self.assertEqual(context["type"], "plain_text")
        self.assertIn("<!channel>", context["text"])
        self.assertNotIn("raw_output", json.dumps(payload))
        self.assertNotIn("requesting a refund", json.dumps(payload))

    async def test_one_successful_request(self):
        calls = []
        payload = self.build()
        def handler(request):
            calls.append(request)
            self.assertEqual(json.loads(request.content), payload)
            return httpx.Response(200, text="ok")
        result = await send_slack_alert(payload, env_file=self.env, transport=httpx.MockTransport(handler))
        self.assertTrue(result["delivered"])
        self.assertEqual(len(calls), 1)

    async def test_http_errors_and_unexpected_body_not_retried(self):
        for status, body in ((400, "secret"), (403, "secret"), (429, "secret"),
                             (500, "secret"), (302, "secret"), (200, "not_ok")):
            calls = []
            def handler(request):
                calls.append(request)
                return httpx.Response(status, text=body, headers={"location": "https://example.com"})
            with self.subTest(status=status), self.assertRaises(SlackAlertError) as error:
                await send_slack_alert(self.build(), env_file=self.env, transport=httpx.MockTransport(handler))
            self.assertNotIn("secret", str(error.exception))
            self.assertEqual(len(calls), 1)

    async def test_transport_failure_redacted_and_no_retry(self):
        for kind in (httpx.ReadTimeout, httpx.ConnectError):
            calls = []
            def handler(request):
                calls.append(request)
                raise kind("secret https://hooks.slack.com/services/TEST/TEST/secret")
            with self.assertRaises(SlackAlertError) as error:
                await send_slack_alert(self.build(), env_file=self.env, transport=httpx.MockTransport(handler))
            self.assertNotIn("secret", str(error.exception))
            self.assertIn("unknown", str(error.exception))
            self.assertEqual(len(calls), 1)

    async def test_missing_or_wrong_webhook_never_sends(self):
        for value in ("", "https://example.com/services/A/B/C", "http://hooks.slack.com/services/A/B/C",
                      "https://hooks.slack.com/services/A/B/C?secret=x"):
            calls = []
            def handler(request):
                calls.append(request)
                return httpx.Response(200, text="ok")
            with patch.dict(os.environ, {"SLACK_WEBHOOK_URL": value}), self.assertRaises(SlackAlertError):
                await send_slack_alert(self.build(), env_file=self.env, transport=httpx.MockTransport(handler))
            self.assertEqual(calls, [])
