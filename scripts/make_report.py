#!/usr/bin/env python3
"""Create a factual CSV summary from evaluation JSONL."""

import argparse
from pathlib import Path

import pandas as pd

from face_destyle.data.metadata import read_jsonl
from face_destyle.schemas import EvaluationRecord


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = read_jsonl(args.evaluations, EvaluationRecord)
    frame = pd.DataFrame([item.model_dump(mode="json") for item in records])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
