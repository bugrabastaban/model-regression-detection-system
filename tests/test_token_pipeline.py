"""Full offline token path: mocked Gemini SDK and intercepted Groq HTTP only."""

from copy import deepcopy
from html.parser import HTMLParser
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

import httpx

from comparison import compare_runs
from drift_detection import detect_slow_drift
from evaluation_runner import run_evaluation
from groq_judge import DEFAULT_MODEL, GroqJudge
from html_report import render_report
from prompt_config import load_prompt_config
from scoring import score_results, save_scored_results


UNKNOWN = {"input_tokens": None, "output_tokens": None, "total_tokens": None}


class _Rows(HTMLParser):
    """Read visible table cells so assertions bind counters to actual case IDs."""

    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = []
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag == "td":
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag == "td" and self.cell is not None:
            self.row.append(''.join(self.cell).strip())
            self.cell = None
        elif tag == "tr":
            self.rows.append(self.row)


class TokenPipelineTests(IsolatedAsyncioTestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        env = patch.dict(os.environ, {"GEMINI_API_KEY": "offline-gemini-key",
                                     "GROQ_API_KEY": "offline-groq-key", "GROQ_MODEL": DEFAULT_MODEL}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        sdk = patch("email_classifier.genai.Client")
        factory = sdk.start()
        self.addCleanup(sdk.stop)
        self.client = factory.return_value.__enter__.return_value
        self.keys = ("full", "missing", "partial", "zero", "bad_metadata", "bad_json", "bad_category", "empty")
        self.dataset = {"dataset_version": "offline-e2e", "cases": [
            {"id": key, "input": f"Support request {key}", "expected_category": "general",
             "ideal_summary": f"The customer requests help with {key}."} for key in self.keys]}
        self.groq_calls = []

        def judge_response(request):
            body = json.loads(request.content)
            email = json.loads(body["messages"][1]["content"])["email"]
            index = next(i for i, case in enumerate(self.dataset["cases"]) if case["input"] == email)
            self.groq_calls.append(email)
            return httpx.Response(200, json={
                "choices": [{"finish_reason": "stop", "message": {
                    "content": '{"score":5,"reason":"Faithful offline summary."}'}}],
                "usage": {"prompt_tokens": 30 + index, "completion_tokens": 3 + index,
                          "total_tokens": 50 + index}})

        self.judge = GroqJudge(env_file=self.directory / "nonexistent.env",
                               transport=httpx.MockTransport(judge_response))

    async def collect(self, metadata, invalid_outputs=None):
        references = {case["input"]: case for case in self.dataset["cases"]}

        def generate(*, contents, **kwargs):
            case = references[contents]
            text = json.dumps({"category": "general", "summary": case["ideal_summary"]})
            if invalid_outputs is not None and case["id"] in invalid_outputs:
                text = invalid_outputs[case["id"]]
            attributes = {"usage_metadata": metadata[case["id"]]} if case["id"] in metadata else {}
            return SimpleNamespace(text=text, **attributes)

        self.client.models.generate_content.side_effect = generate
        return await run_evaluation(load_prompt_config(), self.dataset, batch_size=2)

    @staticmethod
    def without_usage(report):
        # JSON round-trip also represents loading a legacy artifact from disk.
        report = json.loads(json.dumps(report, allow_nan=False))
        for row in report["results"]:
            for container in (row, row.get("scores", {}), row.get("scores", {}).get("judge", {})):
                container.pop("token_usage", None)
                container.pop("token_usage_error", None)
        return report

    async def test_sdk_to_saved_json_comparison_and_html_with_legacy_baseline(self):
        metadata = {key: {"prompt_token_count": 100 + i, "candidates_token_count": i,
                          "total_token_count": 200 + i} for i, key in enumerate(self.keys)}
        del metadata["missing"]
        metadata["partial"] = {"prompt_token_count": 7, "candidates_token_count": 0}
        metadata["zero"] = dict.fromkeys(("prompt_token_count", "candidates_token_count", "total_token_count"), 0)
        metadata["bad_metadata"] = {"prompt_token_count": -1}
        invalid = {"bad_json": "not JSON", "bad_category": '{"category":"other","summary":"Help."}',
                   "empty": None}
        expected = {key: {"input_tokens": 100 + i, "output_tokens": i, "total_tokens": 200 + i}
                    for i, key in enumerate(self.keys)}
        expected.update(missing=UNKNOWN, partial={**UNKNOWN, "input_tokens": 7, "output_tokens": 0},
                        zero=dict.fromkeys(UNKNOWN, 0), bad_metadata=UNKNOWN)
        raw = await self.collect(metadata, invalid)
        before = deepcopy(raw)
        scored = await score_results(raw, self.dataset, judge=self.judge)
        self.assertEqual(raw, before)
        scored.update(run_id="current", started_at="2026-01-02T00:00:00+00:00")
        path = self.directory / "scored.json"
        save_scored_results(scored, path)
        current = json.loads(path.read_text(encoding="utf-8"))
        baseline = self.without_usage(current)
        baseline.update(run_id="legacy", started_at="2026-01-01T00:00:00+00:00")
        diff = compare_runs(current, baseline, current_dataset=self.dataset, previous_dataset=self.dataset)
        html = render_report(current, self.dataset, previous=baseline, previous_dataset=self.dataset)
        html_path = self.directory / "report.html"
        html_path.write_text(html, encoding="utf-8")
        parser = _Rows()
        parser.feed(html_path.read_text(encoding="utf-8"))
        token_rows = {(row[0], row[1], row[2]): row[3:] for row in parser.rows
                      if len(row) == 6 and row[0] in self.keys and row[1] in ("Gemini", "Groq judge")}
        self.assertEqual(len(token_rows), len(self.keys) * 6)
        self.assertEqual(self.client.models.generate_content.call_count, len(self.keys))
        self.assertEqual(len(self.groq_calls), 5)
        self.assertEqual([row["case_id"] for row in current["results"]], list(self.keys))
        for index, (row, entry) in enumerate(zip(current["results"], diff["token_usage_comparisons"])):
            key = row["case_id"]
            counts = expected[key]
            self.assertEqual(entry["case_id"], key)
            self.assertEqual(row["token_usage"], counts)
            self.assertEqual(row["scores"]["token_usage"], counts)
            self.assertEqual(entry["gemini"]["current"], counts)
            self.assertEqual(entry["gemini"]["previous"], UNKNOWN)
            self.assertEqual(entry["gemini"]["delta_tokens"], UNKNOWN)
            judge_counts = UNKNOWN if key in invalid else {
                "input_tokens": 30 + index, "output_tokens": 3 + index, "total_tokens": 50 + index}
            self.assertEqual(row["scores"]["judge"]["token_usage"], judge_counts)
            self.assertEqual(entry["groq"]["current"], judge_counts)
            for provider, usage in (("Gemini", counts), ("Groq judge", judge_counts)):
                for field, label in (("input_tokens", "Input"), ("output_tokens", "Output"),
                                     ("total_tokens", "Total")):
                    visible = "bilinmiyor" if usage[field] is None else str(usage[field])
                    self.assertEqual(token_rows[key, provider, label], ["bilinmiyor", visible, "bilinmiyor"])
            if key in invalid:
                self.assertEqual(row["raw_output"], invalid[key])
                self.assertIsNone(row["output"])
                self.assertEqual(row["error"]["type"], "ValueError")
                self.assertEqual(row["scores"]["judge"]["status"], "skipped")
            else:
                self.assertIsNone(row["error"])
                self.assertEqual(row["scores"]["category_match"], 1)
                self.assertEqual(row["scores"]["summary_relevance"], 5)
        self.assertEqual(current["results"][4]["scores"]["token_usage_error"], "invalid_metadata")
        self.assertEqual(diff["overall_pass_rate"]["compared_cases"], 5)
        self.assertEqual(diff["regressions"], [])

    async def test_metadata_does_not_change_end_to_end_quality_decisions_or_seven_run_drift(self):
        self.dataset["cases"] = self.dataset["cases"][:2]
        enriched_runs, legacy_runs = [], []
        for index in range(7):
            metadata = {"full": {"prompt_token_count": 12, "candidates_token_count": 4,
                                 "total_token_count": 20}, "missing": {"prompt_token_count": 0}}
            wrong = {"full": '{"category":"billing","summary":"The customer requests help."}'} if index == 6 else None
            raw = await self.collect(metadata, wrong)
            raw.update(run_id=f"offline-{index}", started_at=f"2026-01-{index + 1:02d}T00:00:00+00:00")
            enriched = await score_results(raw, self.dataset, judge=self.judge)
            legacy = await score_results(self.without_usage(raw), self.dataset, judge=self.judge)
            for old, new in zip(legacy["results"], enriched["results"]):
                for field in ("category_match", "summary_relevance", "status", "error", "latency_seconds"):
                    self.assertEqual(old["scores"][field], new["scores"][field])
            enriched_runs.append(enriched)
            legacy_runs.append(legacy)

        def compare(runs):
            result = compare_runs(runs[-1], runs[-2], current_dataset=self.dataset, previous_dataset=self.dataset)
            result.pop("token_usage_comparisons")
            return result

        diff = compare(enriched_runs)
        self.assertEqual(diff, compare(legacy_runs))
        self.assertEqual(diff["regressions"], ["full"])
        self.assertEqual(diff["overall_pass_rate"]["current_rate"], 0.5)
        self.assertEqual(diff["threshold_assessment"]["severity"], "critical")

        def drift(runs):
            return detect_slow_drift(runs[-1], self.dataset, history=[(run, self.dataset) for run in runs[:-1]])

        self.assertEqual(drift(enriched_runs), drift(legacy_runs))
        self.assertEqual(drift(enriched_runs)["available_consecutive_runs"], 7)
        for runs in (enriched_runs, legacy_runs):
            html = render_report(runs[-1], self.dataset, previous=runs[-2], previous_dataset=self.dataset,
                                 history=[(run, self.dataset) for run in runs[:-1]])
            self.assertIn("Gerileme: 1", html)
            self.assertIn("50.0%", html)
            self.assertIn("per_run_alert_present", html)
        self.assertEqual(self.client.models.generate_content.call_count, 14)
