"""A small Gemini-powered customer support email classifier."""

import json
import os

from google import genai
from pydantic import ValidationError

from contracts import ClassifierOutput
from prompt_config import load_prompt_config, render_prompt


DEFAULT_PROMPT = render_prompt(load_prompt_config())


def classify_email_raw(
    email: str,
    prompt: str = DEFAULT_PROMPT,
    *,
    model: str = "gemini-3.6-flash",
) -> str | None:
    """Call the LLM feature and preserve its response text before validation."""
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

    return response.text


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
