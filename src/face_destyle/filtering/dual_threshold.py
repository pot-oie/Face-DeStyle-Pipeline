"""Explicit content/style thresholds with an optional identity gate."""

from dataclasses import dataclass

from face_destyle.schemas import EvaluationRecord


@dataclass(frozen=True)
class ThresholdConfig:
    content_threshold: float
    style_removal_threshold: float
    identity_threshold: float = 0.0
    use_identity_threshold: bool = False


def apply_thresholds(record: EvaluationRecord, config: ThresholdConfig) -> EvaluationRecord:
    failures: list[str] = []
    if record.content_score < config.content_threshold:
        failures.append("content_below_threshold")
    if record.style_removal_score < config.style_removal_threshold:
        failures.append("style_removal_below_threshold")
    if config.use_identity_threshold:
        if record.identity_score is None:
            failures.append("identity_score_missing")
        elif record.identity_score < config.identity_threshold:
            failures.append("identity_below_threshold")
    return record.model_copy(
        update={"accepted": not failures, "failure_reason": ";".join(failures) or None}
    )
