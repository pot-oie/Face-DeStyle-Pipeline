#!/usr/bin/env python3
"""Summarize raw formal metrics by method and style without applying thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from face_destyle.data.metadata import read_jsonl
from face_destyle.schemas import FormalEvaluationRecord

METRIC_COLUMNS = [
    "dinov2_cosine",
    "clip_cosine",
    "arcface_cosine",
    "qwen_content_score",
    "qwen_style_removal_score",
    "qwen_identity_score",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records = read_jsonl(args.evaluations, FormalEvaluationRecord)
    if not records:
        raise ValueError("evaluation file is empty")
    frame = pd.DataFrame([record.model_dump(mode="json") for record in records])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "formal-evaluations.csv", index=False)

    grouped = frame.groupby(["method", "style_category"], dropna=False)[METRIC_COLUMNS]
    means = grouped.mean().reset_index()
    counts = grouped.count().reset_index()
    means.to_csv(args.output_dir / "metric-means-by-method-style.csv", index=False)
    counts.to_csv(args.output_dir / "metric-counts-by-method-style.csv", index=False)

    failures: dict[str, int] = {}
    for record in records:
        for metric in record.failures:
            failures[metric] = failures.get(metric, 0) + 1
    summary = {
        "record_count": len(records),
        "methods": sorted(frame["method"].unique().tolist()),
        "styles": sorted(frame["style_category"].unique().tolist()),
        "records_with_any_failure": sum(bool(record.failures) for record in records),
        "failure_counts_by_metric": failures,
        "note": "Raw metrics only; no calibrated thresholds or acceptance claims applied.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Summarized {len(records)} raw evaluations in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
