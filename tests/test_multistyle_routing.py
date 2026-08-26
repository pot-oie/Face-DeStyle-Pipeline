import csv
from pathlib import Path

import pytest

from face_destyle.routing import (
    REVIEW_FIELDS,
    build_review_rows,
    load_review,
    resolve_review,
)
from face_destyle.schemas import DestylizationRecord


def record(source_id: str, style: str = "comic") -> DestylizationRecord:
    return DestylizationRecord(
        id=source_id,
        source_id=source_id,
        input_path=Path(f"/{source_id}-input.png"),
        output_path=Path(f"/{source_id}-output.png"),
        style_category=style,
        backend="test",
        seed=42,
    )


def write_review(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_review_routes_terminal_and_action_records(tmp_path):
    records = [record("comic-a"), record("ink-a", "ink"), record("felt-a", "needle_felt")]
    rows = build_review_rows(records, "stage1")
    rows[0]["decision"] = "accept_stage1"
    rows[1]["decision"] = "run_stage2"
    rows[2]["decision"] = "explicit_failure"
    review = tmp_path / "review.csv"
    write_review(review, rows)

    routes, stage2, origami = resolve_review(
        load_review(review), records, records_path=tmp_path / "records.jsonl"
    )

    assert [route["status"] for route in routes] == [
        "terminal_success",
        "action_required",
        "terminal_failure",
    ]
    assert [item.source_id for item in stage2] == ["ink-a"]
    assert origami == []
    assert routes[1]["terminal_output_path"] is None
    assert routes[0]["parent_records"].endswith("records.jsonl")


def test_origami_v1_requires_origami_stage1(tmp_path):
    records = [record("comic-a")]
    rows = build_review_rows(records, "stage1")
    rows[0]["decision"] = "run_origami_v1"

    with pytest.raises(ValueError, match="Origami Stage 1"):
        resolve_review(rows, records, records_path=tmp_path / "records.jsonl")


def test_review_rejects_unknown_decision(tmp_path):
    rows = build_review_rows([record("comic-a")], "stage1")
    rows[0]["decision"] = "pretend_success"
    review = tmp_path / "review.csv"
    write_review(review, rows)

    with pytest.raises(ValueError, match="invalid or missing decision"):
        load_review(review)


def test_resolver_rejects_edited_output_path(tmp_path):
    records = [record("comic-a")]
    rows = build_review_rows(records, "stage1")
    rows[0]["decision"] = "accept_stage1"
    rows[0]["current_output_path"] = "/different.png"

    with pytest.raises(ValueError, match="current output mismatch"):
        resolve_review(rows, records, records_path=tmp_path / "records.jsonl")
