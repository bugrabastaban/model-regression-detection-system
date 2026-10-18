"""A small Gemini-powered customer support email classifier."""

import json
import os
from dataclasses import dataclass

from google import genai
from pydantic import ValidationError

from app.contracts import ClassifierOutput
from app.prompt_config import load_prompt_config, render_prompt
from app.token_usage import TokenUsageResult, extract_gemini_token_usage


DEFAULT_PROMPT = render_prompt(load_prompt_config())


@dataclass(frozen=True)
class _ClassifierResponse:
    """Per-call text and usage, independent of output validation."""

    text: str | None
    usage: TokenUsageResult


def _classify_email_response(
    email: str,
    prompt: str = DEFAULT_PROMPT,
    *,
    model: str = "gemini-3.6-flash",
) -> _ClassifierResponse:
    """Make one SDK call and retain usage before parsing the response text."""
    for name, value in (("email", email), ("prompt", prompt), ("model", model)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string.")

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set the GEMINI_API_KEY environment variable.")

    schema = ClassifierOutput.model_json_schema()
    with genai.Client(api_key=api_key) as client:
        response = client.models.generate_content(
            model=model,
            contents=email,
            config={
                "system_instruction": prompt,
                "response_mime_type": "application/json",
                "response_json_schema": schema,
            },
        )

    try:
        metadata = getattr(response, "usage_metadata", None)
    except Exception:
        # Reading an SDK property can fail too; never expose its error text.
        usage = TokenUsageResult(error="invalid_metadata")
    else:
        usage = extract_gemini_token_usage(metadata)
    return _ClassifierResponse(text=response.text, usage=usage)


def classify_email_raw(
    email: str,
    prompt: str = DEFAULT_PROMPT,
    *,
    model: str = "gemini-3.6-flash",
) -> str | None:
    """Preserve the public raw-text interface without another SDK request."""
    return _classify_email_response(email, prompt, model=model).text


def parse_classifier_output(text: str | None) -> dict[str, str]:
    """Validate raw model text using the shared output contract."""
    if not text:
        raise ValueError("Gemini returned no text (the request may be blocked).")
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Gemini returned invalid JSON.") from exc
    try:
        validated = ClassifierOutput.model_validate(result)
    except ValidationError as exc:
        raise ValueError("Gemini returned an invalid category or summary.") from exc
    return validated.model_dump()


def classify_email(
    email: str,
    prompt: str = DEFAULT_PROMPT,
    *,
    model: str = "gemini-3.6-flash",
) -> dict[str, str]:
    """Return a validated category/summary dictionary, preserving the public API.

    Invalid input/output raises ValueError; a missing key raises RuntimeError.
    SDK/network errors propagate. One-sentence semantics are prompted, not
    linguistically validated.
    """
    return parse_classifier_output(classify_email_raw(email, prompt, model=model))
