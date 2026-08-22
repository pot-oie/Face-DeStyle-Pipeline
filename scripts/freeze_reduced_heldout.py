#!/usr/bin/env python3
"""Freeze the operator-amended 32-source reduced held-out review."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from face_destyle.reduced_heldout import validate_reduced_review


def build_freeze(review_dir: Path) -> dict[str, object]:
    validated = validate_reduced_review(review_dir)
    selected_sources = sorted(
        (
            {
                "source_id": row["source_id"],
                "style_category": row["style_category"],
                "completed_candidates_at_reduction": int(row["completed_candidates"]),
                "completed_after_reduction": int(row["remaining_candidates"]),
            }
            for row in validated["selected_sources"]
        ),
        key=lambda row: (row["style_category"], row["source_id"]),
    )
    return {
        "schema": "face-destyle-reduced-post-unblinding-freeze/v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_label": "reduced post-unblinding replication analysis",
        "status": "complete_reduced_human_scores_frozen",
        "review_root": str(validated["root"]),
        "input_sha256": validated["input_sha256"],
        "matrix": validated["summary"],
        "selection": {
            "timing": "after method unblinding",
            "source_count": 32,
            "sources_per_style": 8,
            "methods_per_source": 5,
            "selection_priority": "highest existing human-score completion count",
            "tie_break_seed": 20260822,
            "inherited_complete_scores": 99,
            "completed_after_selection": 61,
            "selected_sources": selected_sources,
        },
        "operator_amendment": {
            "original_300_candidate_scoring_abandoned": True,
            "original_60_candidate_repeat_abandoned": True,
            "repeat_reliability_available": False,
            "original_review_materials_preserved": True,
            "original_review_materials_overwritten": False,
        },
        "acceptance_rule": (
            "content_score>=4 and style_removal_score>=4 and "
            "identity_judgment_valid=yes and identity_score>=4"
        ),
        "policies": {
            "identity_unjudgeable": "fail",
            "missing_core": "no imputation",
            "blank_failure_types": "not_reported",
            "composite_score": "not_constructed",
            "test_threshold_selection": "not_permitted",
        },
        "interpretation": (
            "Post-unblinding, completion-informed reduced replication/exploratory analysis; "
            "not an equivalent substitute for the preregistered 300-candidate confirmatory "
            "held-out test."
        ),
    }


def write_new_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing freeze record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--freeze-record", type=Path, required=True)
    parser.add_argument(
        "--external-marker",
        type=Path,
        help="Defaults to REVIEW_DIR/REDUCED_COMPLETE.json.",
    )
    args = parser.parse_args()
    marker = args.external_marker or args.review_dir / "REDUCED_COMPLETE.json"
    try:
        payload = build_freeze(args.review_dir)
        write_new_json(args.freeze_record, payload)
        try:
            write_new_json(marker, payload)
        except Exception:
            args.freeze_record.unlink(missing_ok=True)
            raise
    except (FileExistsError, FileNotFoundError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Frozen reduced review in {args.freeze_record} and {marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
