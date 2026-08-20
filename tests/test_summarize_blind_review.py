import csv
import json
import runpy
import sys
from pathlib import Path


def _write_scores(path: Path, blind_id: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "blind_id",
                "style_category",
                "content_score",
                "style_removal_score",
                "identity_score",
                "identity_judgment_valid",
                "failure_types",
                "reviewer_notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "blind_id": blind_id,
                "style_category": "comic",
                "content_score": "4",
                "style_removal_score": "5",
                "identity_score": "4",
                "identity_judgment_valid": "yes",
                "failure_types": "",
                "reviewer_notes": "ok",
            }
        )


def test_summarize_blind_review_cli(tmp_path: Path, monkeypatch) -> None:
    key_path = tmp_path / "key.jsonl"
    key_rows = [
        {
            "round": round_name,
            "blind_id": f"{round_name.upper()}-0001",
            "canonical_id": "method:source",
            "method": "method",
            "source_id": "source",
            "style_category": "comic",
        }
        for round_name in ("a", "b")
    ]
    key_path.write_text(
        "".join(json.dumps(row) + "\n" for row in key_rows), encoding="utf-8"
    )
    round_a = tmp_path / "a.csv"
    round_b = tmp_path / "b.csv"
    _write_scores(round_a, "A-0001")
    _write_scores(round_b, "B-0001")
    output = tmp_path / "summary"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_blind_review.py",
            "--round-a",
            str(round_a),
            "--round-b",
            str(round_b),
            "--private-key",
            str(key_path),
            "--output-dir",
            str(output),
        ],
    )
    try:
        runpy.run_path("scripts/summarize_blind_review.py", run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0
    result = json.loads((output / "human-summary.json").read_text())
    assert result["unique_candidates"] == 1
    assert result["round_agreement"]["style_removal_score"]["exact_agreement_rate"] == 1.0


def test_summarize_blind_review_accepts_one_complete_round(
    tmp_path: Path, monkeypatch
) -> None:
    key_path = tmp_path / "key.jsonl"
    key_rows = [
        {
            "round": round_name,
            "blind_id": f"{round_name.upper()}-0001",
            "canonical_id": "method:source",
            "method": "method",
            "source_id": "source",
            "style_category": "comic",
        }
        for round_name in ("a", "b")
    ]
    key_path.write_text(
        "".join(json.dumps(row) + "\n" for row in key_rows), encoding="utf-8"
    )
    round_a = tmp_path / "a.csv"
    _write_scores(round_a, "A-0001")
    output = tmp_path / "summary"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_blind_review.py",
            "--round-a",
            str(round_a),
            "--private-key",
            str(key_path),
            "--output-dir",
            str(output),
        ],
    )
    try:
        runpy.run_path("scripts/summarize_blind_review.py", run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0
    result = json.loads((output / "human-summary.json").read_text())
    assert result["scored_rows"] == 1
    assert result["unique_candidates"] == 1
    assert result["round_agreement"]["paired_candidates"] == 0
    assert result["failure_annotation"]["not_reported_rows"] == 1
