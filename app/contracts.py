"""Classifier output contract shared by model calls and prompt examples."""

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


Category = Literal["billing", "technical", "account", "general"]


class ClassifierOutputDict(TypedDict):
    category: Category
    summary: str


class ClassifierOutput(BaseModel):
    """Validated output; one-sentence semantics are enforced by the prompt."""

    model_config = ConfigDict(extra="forbid", strict=True)

    category: Category
    summary: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] = Field(
        description="A concise, one-sentence summary of the email."
    )
