#!/usr/bin/env python3
"""Evaluate pairs with a transparent pixel-similarity smoke metric."""

import argparse
from pathlib import Path

from tqdm import tqdm

from face_destyle.data.metadata import read_jsonl, write_jsonl
from face_destyle.metrics.smoke_test import smoke_test_similarity
from face_destyle.schemas import DestylizationRecord, EvaluationRecord


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--smoke-style-removal-score",
        type=float,
        default=1.0,
        help="Pipeline-test sentinel only; it is not a measured research metric.",
    )
    args = parser.parse_args()
    if not 0 <= args.smoke_style_removal_score <= 1:
        parser.error("--smoke-style-removal-score must be in [0, 1]")
    records = read_jsonl(
        args.records,
        DestylizationRecord,
        check_paths=True,
        path_fields=("input_path", "output_path"),
    )
    evaluations = []
    for record in tqdm(records):
        similarity = smoke_test_similarity(record.input_path, record.output_path)
        evaluations.append(
            EvaluationRecord(
                id=record.id,
                source_id=record.source_id,
                input_path=record.input_path,
                output_path=record.output_path,
                style_category=record.style_category,
                content_score=similarity,
                style_removal_score=args.smoke_style_removal_score,
                smoke_test_similarity=similarity,
                evaluation_mode="smoke_test_with_unmeasured_style_sentinel",
            )
        )
    write_jsonl(evaluations, args.output)
    print("WARNING: smoke_test_similarity is pixel similarity, not DINO, ArcFace, CLIP, or VLM.")
    print(f"Wrote {len(evaluations)} smoke-test evaluations to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
