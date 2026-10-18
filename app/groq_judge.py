"""Groq adapter for scoring's async judge callback (no calls on import)."""

import os
from pathlib import Path

from dotenv import dotenv_values
import httpx
from pydantic import ValidationError

from app.scoring import JudgeResult, JudgeScore, TokenUsage


DEFAULT_MODEL = "openai/gpt-oss-120b"
ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


class GroqJudgeError(RuntimeError):
    """Safe provider error; never includes request headers or response bodies."""


class GroqJudge:
    """Callable judge using existing httpx; one request per call, no retries.

    Environment overrides the project .env. Creating the adapter is offline.
    Calling it sends email and summaries to Groq. A transport can be injected
    for offline tests. Score validation remains in score_results so invalid
    verdict text and valid usage are retained in the scoring report. Malformed
    usage becomes unknown without discarding the verdict or changing its score.
    """

    def __init__(self, *, env_file: str | Path | None = None,
                 transport: httpx.AsyncBaseTransport | None = None):
        values = dotenv_values(env_file if env_file is not None else Path(__file__).resolve().parents[1] / ".env")
        self._api_key = os.environ.get("GROQ_API_KEY", values.get("GROQ_API_KEY") or "").strip()
        self.model = os.environ.get("GROQ_MODEL", values.get("GROQ_MODEL") or DEFAULT_MODEL).strip()
        if not self._api_key:
            raise ValueError("GROQ_API_KEY is not configured.")
        if not self.model:
            raise ValueError("GROQ_MODEL must not be empty.")
        self._transport = transport

    async def __call__(self, system_prompt: str, payload_json: str) -> JudgeResult:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": payload_json}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "summary_relevance", "strict": True,
                "schema": JudgeScore.model_json_schema(),
            }},
            "temperature": 0,
            "max_completion_tokens": 2048,
        }
        try:
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=False,
                                         transport=self._transport) as client:
                response = await client.post(ENDPOINT, json=body,
                                             headers={"Authorization": f"Bearer {self._api_key}"})
        except httpx.TimeoutException:
            raise GroqJudgeError("Groq request timed out; no retry was attempted.") from None
        except httpx.TransportError:
            raise GroqJudgeError("Groq connection failed; no retry was attempted.") from None
        if response.status_code != 200:
            raise GroqJudgeError(f"Groq returned HTTP {response.status_code}; no retry was attempted.")
        try:
            data = response.json()
            choice = data["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("Incomplete response")
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Missing content")
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise GroqJudgeError("Groq returned an invalid or incomplete response envelope.") from None

        # Usage is optional telemetry, independent of the judge's verdict.
        # As with Gemini, discard all counters if any supplied count is invalid.
        usage = data.get("usage")
        token_usage = TokenUsage()
        if isinstance(usage, dict):
            try:
                token_usage = TokenUsage(input_tokens=usage.get("prompt_tokens"),
                                         output_tokens=usage.get("completion_tokens"),
                                         total_tokens=usage.get("total_tokens"))
            except ValidationError:
                pass
        return JudgeResult(raw_output=content, model=self.model, token_usage=token_usage)
