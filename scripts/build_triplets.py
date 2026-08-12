#!/usr/bin/env python3
"""Build deterministic same-style, distinct-source triplets."""

import argparse
from pathlib import Path

from face_destyle.data.metadata import read_jsonl, write_jsonl
from face_destyle.data.triplets import build_triplets
from face_destyle.schemas import EvaluationRecord


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("-k", "--references-per-target", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = read_jsonl(args.evaluations, EvaluationRecord)
    triplets = build_triplets(
        records, references_per_target=args.references_per_target, seed=args.seed
    )
    write_jsonl(triplets, args.output)
    print(f"Wrote {len(triplets)} triplets to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
