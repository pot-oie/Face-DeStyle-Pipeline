#!/usr/bin/env python3
"""Build a strict runtime manifest from a richer private provenance JSONL file."""

import argparse
from collections import defaultdict
from pathlib import Path

from face_destyle.data.metadata import write_jsonl
from face_destyle.data.runtime_manifests import build_runtime_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Rich provenance JSONL input.")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Dataset root containing the anchored relative image paths.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("pilot", "calibration", "test", "extension"),
        default="pilot",
    )
    parser.add_argument(
        "--path-anchor",
        default="raw",
        help="Path component where portable asset paths begin (default: raw).",
    )
    parser.add_argument(
        "--limit-per-style",
        type=int,
        help="Deterministically keep at most this many records per style.",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if args.limit_per_style is not None and args.limit_per_style < 1:
        parser.error("--limit-per-style must be positive")
    records = build_runtime_manifest(
        args.input,
        args.data_root.resolve(),
        split=args.split,
        path_anchor=args.path_anchor,
        limit_per_style=args.limit_per_style,
    )
    if not records:
        parser.error(f"no accepted {args.split!r} records found")
    write_jsonl(records, args.output)
    counts: defaultdict[str, int] = defaultdict(int)
    for record in records:
        counts[record.style_category] += 1
    summary = ", ".join(f"{style}={count}" for style, count in sorted(counts.items()))
    print(f"Wrote {len(records)} runtime records to {args.output} ({summary})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
