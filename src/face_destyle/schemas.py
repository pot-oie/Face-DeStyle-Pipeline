"""Validated record schemas exchanged between pipeline stages."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImageRecord(StrictRecord):
    id: str = Field(min_length=1)
    image_path: Path
    style_category: str = Field(min_length=1)
    source_id: str = Field(min_length=1)


class DatasetManifestRecord(StrictRecord):
    """Portable, checksum-pinned input declaration resolved against a separate data root."""

    id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_group_id: str = Field(min_length=1)
    asset_path: Path
    style_category: str = Field(min_length=1)
    split: Literal["pilot", "calibration", "test", "extension"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qc_status: Literal["accepted"] = "accepted"

    @model_validator(mode="after")
    def validate_asset_path(self) -> "DatasetManifestRecord":
        if self.asset_path.is_absolute() or ".." in self.asset_path.parts:
            raise ValueError("asset_path must be a safe path relative to the data root")
        return self


class DestylizationRecord(StrictRecord):
    id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    input_path: Path
    output_path: Path
    style_category: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    seed: int
    prompt: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class EvaluationRecord(StrictRecord):
    id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    input_path: Path
    output_path: Path
    style_category: str = Field(min_length=1)
    content_score: float = Field(ge=0.0, le=1.0)
    style_removal_score: float = Field(ge=0.0, le=1.0)
    identity_score: float | None = Field(default=None, ge=0.0, le=1.0)
    smoke_test_similarity: float = Field(ge=0.0, le=1.0)
    evaluation_mode: str = "smoke_test"
    accepted: bool | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> "EvaluationRecord":
        if self.accepted is True and self.failure_reason:
            raise ValueError("accepted records cannot have a failure_reason")
        return self


class FormalEvaluationRecord(StrictRecord):
    """Raw formal metric outputs before human calibration and thresholding."""

    id: str = Field(min_length=1)
    record_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    method: str = Field(min_length=1)
    input_path: Path
    output_path: Path
    style_category: str = Field(min_length=1)
    dinov2_cosine: float | None = Field(default=None, ge=-1.0, le=1.0)
    clip_cosine: float | None = Field(default=None, ge=-1.0, le=1.0)
    arcface_cosine: float | None = Field(default=None, ge=-1.0, le=1.0)
    arcface_status: str | None = None
    qwen_content_score: int | None = Field(default=None, ge=0, le=5)
    qwen_style_removal_score: int | None = Field(default=None, ge=0, le=5)
    qwen_identity_score: int | None = Field(default=None, ge=0, le=5)
    qwen_evidence: str | None = None
    qwen_raw_response: str | None = None
    failures: dict[str, str] = Field(default_factory=dict)
    evaluator_metadata: dict[str, Any] = Field(default_factory=dict)


class TripletRecord(StrictRecord):
    id: str = Field(min_length=1)
    destylized_content_path: Path
    style_reference_path: Path
    original_style_target_path: Path
    style_category: str = Field(min_length=1)
    target_source_id: str = Field(min_length=1)
    reference_source_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def ensure_distinct_sources(self) -> "TripletRecord":
        if self.target_source_id == self.reference_source_id:
            raise ValueError("reference and target must have different source_id values")
        return self
