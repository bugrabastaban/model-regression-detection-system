"""CI orchestration tests with mocked providers and local git/data only."""

from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

import yaml

import ci_pipeline as ci
from prompt_config import load_prompt_config
from scoring import JudgeResult


ROOT = Path(__file__).resolve().parents[1]


class CIPipelineTests(IsolatedAsyncioTestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name)
        self.config = ci.read_json(ROOT / "ci/config.json")
        self.dataset = ci.read_json(ROOT / "datasets/golden_v1.json")
        self.baseline = ci.read_json(ROOT / "ci/baseline.json")
        self.snapshot = ci.read_json(ROOT / "ci/baseline_dataset.json")
        self.plan = {"prompt_version": "v1", "prompts_dir": str(ROOT / "prompts"),
                     "head_sha": "b" * 40, "base_sha": "a" * 40}
        ci.write_json(self.output / "plan.json", self.plan)

    async def test_fixed_baseline_is_complete_and_prompt_matches(self):
        cases = ci.validate_subset(self.config, self.dataset, self.baseline, self.snapshot)
        self.assertEqual([case["id"] for case in cases], self.config["case_ids"])
        self.assertEqual(ci.prompt_digest(load_prompt_config("v1", prompts_dir=ROOT / self.config["baseline_prompts_dir"])),
                         self.config["baseline_prompt_sha256"])

    async def test_preflight_rejects_missing_changed_and_expanded_cases(self):
        for kind in ("expanded", "changed", "incomplete", "interval"):
            config, dataset, baseline = deepcopy(self.config), deepcopy(self.dataset), deepcopy(self.baseline)
            if kind == "expanded": config["case_ids"].append("email_003")
            if kind == "changed": dataset["cases"][0]["input"] = "Changed"
            if kind == "incomplete": baseline["results"][0]["error"] = {"type": "TimeoutError"}
            if kind == "interval": config["request_interval_seconds"] = 0
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                ci.validate_subset(config, dataset, baseline, self.snapshot)

    async def test_no_prompt_change_is_applicable_without_quota(self):
        with patch("ci_pipeline.git", return_value="") as git:
            state = ci.prepare(self.output, "a" * 40, "b" * 40)
        self.assertEqual(state["status"], "NOT_APPLICABLE")
        self.assertEqual(git.call_count, 1)

    async def test_fork_prompt_changes_block_before_reading_head(self):
        with patch.dict(os.environ, {"CI_SAME_REPOSITORY": "false"}), patch(
            "ci_pipeline.git", return_value="prompts/v1.yaml\n"
        ) as git:
            state = ci.prepare(self.output, "a" * 40, "b" * 40)
        self.assertEqual(state["status"], "INCOMPLETE")
        self.assertEqual(git.call_count, 1)

    async def test_candidate_loads_highest_yaml_version_only(self):
        config = asdict(load_prompt_config("v1"))
        config["version_id"] = "v2"
        tree = "100644 blob " + "0" * 40 + "\tprompts/v2.yaml\n"
        responses = ["prompts/v2.yaml\n", tree, yaml.safe_dump(config)]
        with patch.dict(os.environ, {"CI_SAME_REPOSITORY": "true"}), patch(
            "ci_pipeline.git", side_effect=responses
        ):
            state = ci.prepare(self.output, "a" * 40, "b" * 40)
        self.assertEqual(state["status"], "READY")
        self.assertEqual(state["plan"]["prompt_version"], "v2")

    async def test_symlink_and_bad_sha_rejected(self):
        with self.assertRaises(ValueError):
            ci.prepare(self.output, "--bad", "b" * 40)
        with patch.dict(os.environ, {"CI_SAME_REPOSITORY": "true"}), patch(
            "ci_pipeline.git", side_effect=["prompts/v2.yaml", "120000 blob " + "0" * 40 + "\tprompts/v2.yaml"]
        ), self.assertRaises(ValueError):
            ci.prepare(self.output, "a" * 40, "b" * 40)

    async def fake_run(self, prompt, subset, **kwargs):
        case = subset["cases"][0]
        return {"dataset_version": subset["dataset_version"], "model": kwargs["model"],
                "prompt_version": prompt.version_id, "started_at": "2026-10-15T10:00:00+00:00",
                "results": [{"case_id": case["id"], "raw_output": "{}", "error": None,
                             "elapsed_seconds": 0.1,
                             "output": {"category": case["expected_category"], "summary": case["ideal_summary"]}}]}

    async def evaluate_mocked(self, run=None, verdict=5):
        sdk = patch("google.genai.Client")
        factory = sdk.start()
        self.addCleanup(sdk.stop)
        client = factory.return_value.__enter__.return_value
        client._api_client.vertexai = False
        client._api_client._retry.stop.max_attempt_number = 1
        judge = AsyncMock(return_value=JudgeResult(raw_output=json.dumps({"score": verdict, "reason": "Offline."}),
                                                  model=self.config["judge_model"]))
        judge.model = self.config["judge_model"]
        with patch.dict(os.environ, {"GEMINI_API_KEY": "offline"}), patch(
            "ci_pipeline.GroqJudge", return_value=judge
        ), patch("ci_pipeline.run_evaluation", side_effect=run or self.fake_run) as runner, patch(
            "ci_pipeline.asyncio.sleep", new_callable=AsyncMock
        ) as sleep:
            state = await ci.evaluate(self.output)
        return state, runner, judge, sleep

    async def test_success_three_calls_spacing_and_artifacts(self):
        state, runner, judge, sleep = await self.evaluate_mocked()
        self.assertEqual(state["status"], "PASS")
        self.assertEqual(runner.await_count, 3)
        self.assertEqual(judge.await_count, 3)
        self.assertEqual(sleep.await_count, 2)
        self.assertTrue(all(call.args == (60,) for call in sleep.call_args_list))
        self.assertTrue((self.output / "report.html").exists())
        self.assertEqual(len(ci.read_json(self.output / "scored.json")["results"]), 3)
        self.assertTrue(all(call.kwargs["batch_size"] == 1 for call in runner.call_args_list))

    async def test_critical_blocks_and_summary_has_regressions(self):
        state, *_ = await self.evaluate_mocked(verdict=1)
        self.assertEqual(state["status"], "FAIL")
        summary = ci.summary_markdown(state, "https://github.com/example/repo/actions/runs/1/artifacts/2")
        self.assertIn("Regressions: **3**", summary)
        self.assertIn("100.0% → 0.0%", summary)
        self.assertIn("Download HTML", summary)

    async def test_failure_stops_without_retry_and_redacts(self):
        async def failed(*args, **kwargs):
            result = await self.fake_run(*args, **kwargs)
            result["results"][0].update(output=None, error={"type": "TimeoutError", "message": "secret", "traceback": "secret"})
            return result
        state, runner, judge, sleep = await self.evaluate_mocked(run=failed)
        self.assertEqual(state["status"], "INCOMPLETE")
        self.assertEqual(runner.await_count, 1)
        judge.assert_not_awaited()
        sleep.assert_not_awaited()
        self.assertNotIn("secret", (self.output / "scored.json").read_text())

    async def test_gate_and_finalize_fail_closed(self):
        for status, code in (("PASS", 0), ("WARN", 0), ("NOT_APPLICABLE", 0), ("FAIL", 1), ("INCOMPLETE", 1), ("READY", 1)):
            ci.save_state(self.output, status, "Offline")
            with patch("sys.argv", ["ci_pipeline.py", "gate", "--output", str(self.output)]), self.assertRaises(SystemExit) as exit:
                ci.main()
            self.assertEqual(exit.exception.code, code)
        with patch("sys.argv", ["ci_pipeline.py", "finalize", "--output", str(self.output)]):
            ci.main()
        self.assertEqual(ci.read_json(self.output / "status.json")["status"], "INCOMPLETE")

    async def test_workflow_static_contract(self):
        workflow = yaml.load((ROOT / ".github/workflows/prompt-evaluation.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        self.assertIn("pull_request", workflow["on"])
        self.assertNotIn("pull_request_target", workflow["on"])
        job = workflow["jobs"]["evaluate"]
        self.assertEqual(job["permissions"]["pull-requests"], "write")
        self.assertEqual(job["concurrency"]["cancel-in-progress"], "false")
        for steps in workflow["jobs"].values():
            for step in steps["steps"]:
                if "uses" in step:
                    self.assertRegex(step["uses"], r"@[0-9a-f]{40}$")

    @unittest.skipUnless(shutil.which("node"), "Node.js unavailable")
    async def test_quota_reservations_offline(self):
        script = """
const check = require(process.argv[1]);
const artifacts = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const outputs = {};
check({github:{paginate:async()=>artifacts,rest:{actions:{listArtifactsForRepo(){}}}},
context:{repo:{owner:'test',repo:'test'},runId:123},
core:{setOutput:(k,v)=>outputs[k]=v,info:()=>{}}}).then(()=>console.log(JSON.stringify(outputs))).catch(()=>process.exit(2));
"""
        scenarios = [([], "3", "true"),
                     ([{"name": "gemini-reservation-1-1", "created_at": "2099-01-01T00:00:00Z"}], "3", "false"),
                     ([{"name": "gemini-reservation-1-1", "created_at": "2000-01-01T00:00:00Z"}], "3", "true"),
                     ([], "0", "false")]
        for artifacts, budget, allowed in scenarios:
            with self.subTest(budget=budget, artifacts=artifacts):
                process = subprocess.run([shutil.which("node"), "-e", script, str(ROOT / "ci/quota.cjs")],
                                         input=json.dumps(artifacts), text=True, capture_output=True,
                                         cwd=self.output, env={**os.environ, "CI_DAILY_BUDGET": budget}, timeout=10)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertEqual(json.loads(process.stdout)["allowed"], allowed)
