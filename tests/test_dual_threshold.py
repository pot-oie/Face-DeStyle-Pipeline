from face_destyle.filtering.dual_threshold import ThresholdConfig, apply_thresholds
from face_destyle.schemas import EvaluationRecord


def record(**updates):
    values = {
        "id": "x",
        "source_id": "x",
        "input_path": "input.png",
        "output_path": "output.png",
        "style_category": "comic",
        "content_score": 0.9,
        "style_removal_score": 0.8,
        "identity_score": 0.7,
        "smoke_test_similarity": 1.0,
    }
    values.update(updates)
    return EvaluationRecord(**values)


def test_dual_threshold_accepts_passing_record():
    result = apply_thresholds(record(), ThresholdConfig(0.75, 0.6))
    assert result.accepted is True
    assert result.failure_reason is None


def test_failures_are_explicit_and_identity_is_optional():
    config = ThresholdConfig(0.75, 0.6, identity_threshold=0.8, use_identity_threshold=True)
    result = apply_thresholds(record(content_score=0.4, identity_score=None), config)
    assert result.accepted is False
    assert result.failure_reason == "content_below_threshold;identity_score_missing"
