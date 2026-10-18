"""Minimal HTTP interface for the existing email classifier."""

import os
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google.genai.errors import APIError
import httpx
from pydantic import BaseModel, ConfigDict, StringConstraints

from app.contracts import ClassifierOutput
from app.email_classifier import DEFAULT_PROMPT, classify_email


load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
app = FastAPI(title="Model Regression Detection System")


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    email: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


@app.post("/classify", response_model=ClassifierOutput)
def classify(request: ClassifyRequest) -> ClassifierOutput:
    """Classify one email using the configured Gemini model and YAML prompt."""
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        raise HTTPException(503, "Classifier is not configured.")
    options = {}
    model = os.environ.get("GEMINI_MODEL")
    if model is not None:
        if not model.strip():
            raise HTTPException(503, "Classifier model is not configured.")
        options["model"] = model.strip()
    try:
        # DEFAULT_PROMPT already uses load_prompt_config/render_prompt.
        output = classify_email(request.email, prompt=DEFAULT_PROMPT, **options)
        return ClassifierOutput.model_validate(output)
    except (httpx.TimeoutException, TimeoutError):
        raise HTTPException(504, "Model provider timed out.") from None
    except APIError as exc:
        if exc.code == 429:
            raise HTTPException(429, "Model provider quota or rate limit reached.") from None
        if exc.code in (408, 504):
            raise HTTPException(504, "Model provider timed out.") from None
        raise HTTPException(502, "Model provider request failed.") from None
    except (httpx.TransportError, ConnectionError):
        raise HTTPException(502, "Could not reach the model provider.") from None
    except ValueError:
        raise HTTPException(502, "Model provider returned an invalid response.") from None
    except Exception:
        raise HTTPException(500, "Classification failed.") from None
