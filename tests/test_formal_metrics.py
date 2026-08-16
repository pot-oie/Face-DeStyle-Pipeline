from pathlib import Path

import pytest

from face_destyle.data.metadata import write_jsonl
from face_destyle.metrics.formal import apply_scalar_metric, build_formal_records
from face_destyle.metrics.style_removal import QwenPairRubric
from face_destyle.schemas import DestylizationRecord, FormalEvaluationRecord


def _record(tmp_path: Path) -> DestylizationRecord:
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    source.write_bytes(b"source")
    output.write_bytes(b"output")
    return DestylizationRecord(
        id="sample-1",
        source_id="source-1",
        input_path=source,
        output_path=output,
        style_category="comic",
        backend="mock",
        seed=42,
    )


def test_build_formal_records_requires_method_spec(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="METHOD"):
        build_formal_records([str(tmp_path / "records.jsonl")])


def test_build_and_checkpoint_scalar_metric(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    write_jsonl([_record(tmp_path)], records_path)
    records = build_formal_records([f"adaptive={records_path}"])
    assert records[0].id == "adaptive:sample-1"

    destination = tmp_path / "formal.jsonl"
    apply_scalar_metric(
        records,
        metric_name="dino",
        score_field="dinov2_cosine",
        scorer=lambda _source, _output: 0.75,
        output=destination,
    )
    parsed = FormalEvaluationRecord.model_validate_json(destination.read_text())
    assert parsed.dinov2_cosine == 0.75


def test_scalar_metric_checkpoints_failure(tmp_path: Path) -> None:
    record = FormalEvaluationRecord(
        id="method:one",
        record_id="one",
        source_id="source",
        method="method",
        input_path=tmp_path / "in.png",
        output_path=tmp_path / "out.png",
        style_category="ink",
    )
    destination = tmp_path / "formal.jsonl"

    def fail(_source: Path, _output: Path) -> float:
        raise RuntimeError("expected")

    apply_scalar_metric(
        [record],
        metric_name="clip",
        score_field="clip_cosine",
        scorer=fail,
        output=destination,
    )
    parsed = FormalEvaluationRecord.model_validate_json(destination.read_text())
    assert parsed.clip_cosine is None
    assert parsed.failures["clip"] == "RuntimeError: expected"


def test_qwen_rubric_parses_json_without_loading_model() -> None:
    parsed = QwenPairRubric._parse_json(
        'prefix {"content_preservation": 4, "style_removal": 5, '
        '"identity_preservation": 3, "evidence": "visible note"} suffix'
    )
    assert parsed["style_removal"] == 5


def test_qwen_rubric_rejects_out_of_range_score() -> None:
    with pytest.raises(ValueError, match="integer in"):
        QwenPairRubric._parse_json(
            '{"content_preservation": 6, "style_removal": 5, '
            '"identity_preservation": 3, "evidence": "note"}'
        )
