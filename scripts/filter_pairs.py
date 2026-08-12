#!/usr/bin/env python3
"""Apply content/style and optional identity thresholds."""

import argparse
from pathlib import Path

from face_destyle.data.metadata import read_jsonl, write_jsonl
from face_destyle.filtering.dual_threshold import ThresholdConfig, apply_thresholds
from face_destyle.schemas import EvaluationRecord
from face_destyle.utils.io import load_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation.yaml"))
    args = parser.parse_args()
    raw = load_yaml(args.config)
    config = ThresholdConfig(**raw)
    records = read_jsonl(args.evaluations, EvaluationRecord)
    filtered = [apply_thresholds(record, config) for record in records]
    write_jsonl(filtered, args.output)
    accepted = sum(record.accepted is True for record in filtered)
    print(f"Accepted {accepted}/{len(filtered)} records; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
