"""Bounded asynchronous collection of model outputs; no scoring or persistence."""

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from time import perf_counter
import traceback

from app.email_classifier import _classify_email_response, parse_classifier_output
from app.prompt_config import PromptConfig, render_prompt
from app.token_usage import TokenUsageResult


async def run_evaluation(
    prompt_config: PromptConfig,
    golden_dataset: dict,
    *,
    model: str = "gemini-3.6-flash",
    batch_size: int = 4,
) -> dict:
    """Run each case once in bounded batches and return JSON-serializable results.

    Accepts the loaded golden JSON object. Blocking feature calls run in worker
    threads; each batch is awaited before the next begins. No runner retries or
    scoring. Provider SDK retry behavior still applies. Concurrency reduces wait
    time, not token prices; this is not the provider's discounted Batch API.
    Errors are collected per case, and original response text is retained even
    when output validation fails. Usage is captured before output validation;
    token_usage_error is separate from classification error, and missing counts
    remain null even on API failures. Cancellation propagates; already running
    thread calls cannot be cancelled and may still incur charges.
    """
    if not isinstance(prompt_config, PromptConfig):
        raise TypeError("prompt_config must be a PromptConfig.")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string.")
    if not isinstance(golden_dataset, dict):
        raise ValueError("golden_dataset must be a loaded JSON object.")
    dataset = deepcopy(golden_dataset)
    version = dataset.get("dataset_version")
    cases = dataset.get("cases")
    if not isinstance(version, str) or not version.strip() or not isinstance(cases, list):
        raise ValueError("Dataset requires dataset_version and a cases list.")
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each case must be an object.")
        for field in ("id", "input"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f"Each case requires a non-empty {field}.")
        if case["id"] in seen:
            raise ValueError(f"Duplicate case ID: {case['id']}")
        seen.add(case["id"])
    prompt = render_prompt(prompt_config)
    if not prompt.strip():
        raise ValueError("Rendered prompt must not be empty.")

    async def run_case(case: dict) -> dict:
        start = perf_counter()
        raw_output = None
        output = None
        error = None
        usage = TokenUsageResult()
        try:
            response = await asyncio.to_thread(
                _classify_email_response, case["input"], prompt=prompt, model=model
            )
            raw_output = response.text
            usage = response.usage
            output = parse_classifier_output(raw_output)
        except Exception as exc:
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                # Include chained JSON/Pydantic causes without capturing locals.
                "traceback": "".join(traceback.format_exception(exc)),
            }
        return {
            "case_id": case["id"],
            "raw_output": raw_output,
            "output": output,
            "error": error,
            "elapsed_seconds": perf_counter() - start,
            "token_usage": usage.token_usage.model_dump(),
            "token_usage_error": usage.error,
        }

    report = {
        "prompt_version": prompt_config.version_id,
        "prompt_timestamp": prompt_config.timestamp,
        "dataset_version": version,
        "model": model,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "results": [],
    }
    for offset in range(0, len(cases), batch_size):
        results = await asyncio.gather(*(run_case(case) for case in cases[offset:offset + batch_size]))
        report["results"].extend(results)
    return report
