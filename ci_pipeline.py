"""Quota-conscious PR evaluation using trusted base code and PR prompt data."""

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import os
from pathlib import Path
import re
import subprocess

from comparison import compare_runs
from evaluation_runner import run_evaluation
from groq_judge import GroqJudge
from html_report import render_report
from prompt_config import load_prompt_config
from scoring import score_results


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prompt_digest(prompt):
    return hashlib.sha256(json.dumps(asdict(prompt), sort_keys=True, ensure_ascii=True,
                                     separators=(",", ":")).encode()).hexdigest()


def summary_markdown(state, artifact_url=None):
    """Comment uses bounded, generated fields; never embeds provider messages."""
    lines = ["<!-- model-regression-eval -->", "## Model evaluation: " + state["status"],
             state.get("message", ""), "", "Scope: fixed 3-case subset; not the full 92-case evaluation."]
    comparison = state.get("comparison")
    if comparison:
        metric = comparison["overall_pass_rate"]
        def rate(value):
            return "N/A" if value is None else f"{value * 100:.1f}%"
        delta = metric["delta_percentage_points"]
        lines.extend([
            f"Pass rate: **{rate(metric['previous_rate'])} → {rate(metric['current_rate'])}** "
            f"({delta:+.2f} percentage points)." if delta is not None else "Pass delta unavailable.",
            f"Regressions: **{len(comparison['regressions'])}**; improvements: **{len(comparison['improvements'])}**; "
            f"paired cases: **{metric['compared_cases']}**.",
            f"Threshold severity: **{comparison['threshold_assessment']['severity']}**.",
        ])
    if artifact_url:
        lines.append(f"[Download HTML diff report and JSON results]({artifact_url}) (GitHub access required).")
    else:
        lines.append("Report artifact unavailable; see workflow logs.")
    return "\n\n".join(lines) + "\n"


def save_state(output, status, message, **extra):
    """Refresh the status banner while retaining any generated diff report."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    state = {"status": status, "message": message, **extra}
    write_json(output / "status.json", state)
    (output / "summary.md").write_text(summary_markdown(state), encoding="utf-8")
    report_path = output / "report.html"
    start, end = "<!-- ci-status:start -->", "<!-- ci-status:end -->"
    banner = (f'{start}<section aria-label="CI evaluation status">'
              f'<h2>{escape(status)}</h2><p>{escape(message)}</p></section>{end}')
    html = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    if start in html and end in html:
        before, _, rest = html.partition(start)
        _, _, after = rest.partition(end)
        html = before + banner + after
    elif not html or "<title>Evaluation status</title>" in html:
        # Replace placeholders produced by older versions as well.
        html = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                f'<title>Evaluation status</title></head><body>{banner}</body></html>')
    else:
        body = re.search(r"<body\b[^>]*>", html, re.IGNORECASE)
        position = body.end() if body else 0
        html = html[:position] + banner + html[position:]
    report_path.write_text(html, encoding="utf-8")
    return state


def git(*args):
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8")


def prepare(output, base_sha, head_sha):
    """Extract only regular versioned YAML blobs; never execute PR Python code."""
    for sha in (base_sha, head_sha):
        if not re.fullmatch(r"[0-9a-f]{40,64}", sha):
            raise ValueError("Expected full commit SHA.")
    changed = git("diff", "--name-only", base_sha, head_sha, "--", "prompts/").splitlines()
    if not changed:
        return save_state(output, "NOT_APPLICABLE", "No prompt changes; no API quota used.")
    if os.environ.get("CI_SAME_REPOSITORY") != "true":
        return save_state(output, "INCOMPLETE", "Live evaluation is not authorized for fork PRs; no provider secrets used.")
    output = Path(output)
    prompts_dir = output / "candidate_prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    versions = []
    for line in git("ls-tree", "-r", head_sha, "--", "prompts/").splitlines():
        metadata, name = line.split("\t", 1)
        match = re.fullmatch(r"prompts/v([1-9][0-9]*)\.yaml", name)
        if match:
            mode, kind, _ = metadata.split()
            if mode not in ("100644", "100755") or kind != "blob":
                raise ValueError("Prompt versions must be regular YAML files.")
            text = git("show", f"{head_sha}:{name}")
            if len(text.encode("utf-8")) > 100_000:
                raise ValueError("Prompt file too large for this limited CI check.")
            (prompts_dir / Path(name).name).write_text(text, encoding="utf-8")
            load_prompt_config(f"v{match[1]}", prompts_dir=prompts_dir)
            versions.append(int(match[1]))
    if not versions:
        raise ValueError("No valid versioned YAML prompt remains in the PR.")
    config = read_json("ci/config.json")
    baseline_prompt = load_prompt_config(config["baseline_prompt_version"],
                                         prompts_dir=Path(config["baseline_prompts_dir"]))
    if prompt_digest(baseline_prompt) != config["baseline_prompt_sha256"]:
        raise ValueError("Trusted baseline prompt changed; rebaseline before evaluating this PR.")
    version = f"v{max(versions)}"
    plan = {"prompt_version": version, "prompts_dir": str(prompts_dir),
            "head_sha": head_sha, "base_sha": base_sha}
    write_json(output / "plan.json", plan)
    return save_state(output, "READY", "Prompt changes detected; preflight complete.", plan=plan)


def validate_subset(config, dataset, baseline, snapshot):
    ids = config["case_ids"]
    if len(ids) != 3 or len(set(ids)) != 3 or config["max_gemini_requests"] != 3:
        raise ValueError("CI is limited to exactly three fixed cases.")
    if config["request_interval_seconds"] < 60:
        raise ValueError("CI requires at least 60 seconds between cases.")
    refs = {case["id"]: case for case in dataset["cases"]}
    if len(refs) != len(dataset["cases"]) or not set(ids).issubset(refs):
        raise ValueError("Dataset IDs must be unique and include the CI cohort.")
    if set(ids) != {row["case_id"] for row in baseline["results"]}:
        raise ValueError("Baseline must contain exactly the CI cohort.")
    expected = {case["id"]: case for case in snapshot["cases"]}
    for key in ids:
        if any(refs[key][field] != expected[key][field] for field in ("input", "expected_category", "ideal_summary")):
            raise ValueError("CI references changed; create an approved baseline first.")
    if baseline["model"] != config["model"]:
        raise ValueError("CI model differs from approved baseline.")
    baseline_check = compare_runs(baseline, baseline, current_dataset=snapshot, previous_dataset=snapshot)
    if baseline_check["overall_pass_rate"]["compared_cases"] != 3:
        raise ValueError("Approved baseline must have three complete, comparable scores.")
    return [refs[key] for key in ids]


def _sanitize_report(report):
    """Provider stack traces/bodies must not enter CI artifacts or comments."""
    for row in report["results"]:
        for container, field in ((row, "error"), (row["scores"], "error"), (row["scores"]["judge"], "error")):
            if container.get(field):
                container[field] = {"type": container[field]["type"],
                                    "message": "Provider or output validation failed; evaluation is incomplete."}


async def evaluate(output):
    """Evaluate once per case, serially; stop on the first provider/validation error."""
    output = Path(output)
    plan = read_json(output / "plan.json")
    config = read_json("ci/config.json")
    dataset = read_json("datasets/golden_v1.json")
    baseline = read_json(config["baseline_report"])
    snapshot = read_json(config["baseline_dataset"])
    cases = validate_subset(config, dataset, baseline, snapshot)
    prompt = load_prompt_config(plan["prompt_version"], prompts_dir=Path(plan["prompts_dir"]))
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        raise ValueError("GEMINI_API_KEY is missing.")
    judge = GroqJudge(env_file=output / "nonexistent.env")
    if judge.model != config["judge_model"] or any(
        row["scores"]["judge"]["model"] != judge.model for row in baseline["results"]
    ):
        raise ValueError("Judge configuration differs from baseline.")
    # These are local checks, not an API call. Reject future SDK retry changes.
    from google import genai
    with genai.Client(api_key=os.environ["GEMINI_API_KEY"]) as client:
        if client._api_client.vertexai or client._api_client._retry.stop.max_attempt_number != 1:
            raise ValueError("SDK backend/retry configuration is unsafe for this quota.")
    current = None
    for index, case in enumerate(cases):
        if index:
            await asyncio.sleep(config["request_interval_seconds"])
        subset = {**dataset, "cases": [case]}
        raw = await run_evaluation(prompt, subset, model=config["model"], batch_size=1)
        scored = await score_results(raw, subset, judge=judge)
        _sanitize_report(scored)
        if current is None:
            current = {**scored, "results": [],
                       "run_id": f"github-{os.getenv('GITHUB_RUN_ID', 'local')}-{os.getenv('GITHUB_RUN_ATTEMPT', '1')}",
                       "commit_sha": plan["head_sha"]}
        current["results"].extend(scored["results"])
        row = scored["results"][0]
        if row["error"] or row["scores"]["judge"]["error"]:
            break
    write_json(output / "scored.json", current)
    write_json(output / "dataset_snapshot.json", dataset)
    diff = compare_runs(current, baseline, current_dataset=dataset, previous_dataset=snapshot)
    write_json(output / "comparison.json", diff)
    (output / "report.html").write_text(render_report(current, dataset, previous=baseline,
                                                        previous_dataset=snapshot), encoding="utf-8")
    complete = len(current["results"]) == 3 and diff["overall_pass_rate"]["compared_cases"] == 3
    if not complete:
        return save_state(output, "INCOMPLETE", "Provider error or invalid result; merge check cannot pass.", comparison=diff)
    severity = diff["threshold_assessment"]["severity"]
    status = {"ok": "PASS", "warning": "WARN", "critical": "FAIL"}.get(severity, "INCOMPLETE")
    return save_state(output, status, "Compared PR prompt against the fixed approved subset baseline.", comparison=diff)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "evaluate", "mark", "finalize", "gate"))
    parser.add_argument("--output", type=Path, default=Path("ci_output"))
    parser.add_argument("--base-sha")
    parser.add_argument("--head-sha")
    parser.add_argument("--message", default="Evaluation did not complete.")
    args = parser.parse_args()
    if args.command == "gate":
        state = read_json(args.output / "status.json")
        raise SystemExit(0 if state["status"] in ("PASS", "WARN", "NOT_APPLICABLE") else 1)
    try:
        if args.command == "prepare":
            state = prepare(args.output, args.base_sha, args.head_sha)
        elif args.command == "evaluate":
            state = asyncio.run(evaluate(args.output))
        elif args.command == "finalize":
            path = args.output / "status.json"
            state = read_json(path) if path.exists() else {"status": "READY"}
            if state["status"] == "READY":
                state = save_state(args.output, "INCOMPLETE", "Workflow stopped before evaluation completed; no passing result is available.")
        else:
            state = save_state(args.output, "INCOMPLETE", args.message)
    except Exception:
        state = save_state(args.output, "INCOMPLETE", "Preflight or execution failed; no automatic retry. Check configuration and offline tests.")
    print(state["status"])
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as file:
            file.write(f"ready={'true' if state['status'] == 'READY' else 'false'}\n")


if __name__ == "__main__":
    main()
