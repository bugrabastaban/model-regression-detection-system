"""Shared token contracts and offline Gemini usage extraction; no model calls."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TokenUsage(BaseModel):
    """Unknown counts stay null; provider totals need not equal input + output."""

    model_config = ConfigDict(extra="forbid", strict=True)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


def gemini_token_usage(metadata: object | None) -> TokenUsage:
    """Extract supplied SDK metadata, retaining the legacy strict validation.

    Accept a dictionary or an object with SDK-style attributes. Missing fields
    stay unknown; no counts are estimated. Invalid counts raise ValidationError.
    Use extract_gemini_token_usage at a non-critical metadata boundary instead
    when extraction errors must be separate from classification errors.
    """
    def value(name):
        return metadata.get(name) if isinstance(metadata, dict) else getattr(metadata, name, None)
    return TokenUsage(input_tokens=value("prompt_token_count"),
                      output_tokens=value("candidates_token_count"),
                      total_tokens=value("total_token_count"))


class TokenUsageResult(BaseModel):
    """Usage and a separate, safe metadata error; never a classifier verdict.

    Missing metadata is not an error. If extraction fails, all counts are null
    and error is invalid_metadata. No provider values or exception text are
    exposed. Callers must not treat this error as a classification failure.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    error: Literal["invalid_metadata"] | None = None


def extract_gemini_token_usage(metadata: object | None) -> TokenUsageResult:
    """Non-throwing metadata extraction for future classifier integration.

    Discard the entire usage object on extraction failure rather than silently
    repairing malformed counters. Valid partial metadata is retained. Catch
    ordinary exceptions from SDK attributes as well as counter validation;
    process-control exceptions (BaseException) still propagate.
    """
    try:
        return TokenUsageResult(token_usage=gemini_token_usage(metadata))
    except Exception:
        return TokenUsageResult(error="invalid_metadata")
