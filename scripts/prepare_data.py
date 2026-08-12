#!/usr/bin/env python3
"""Scan an image directory and create validated JSONL metadata."""

import argparse
import hashlib
from pathlib import Path

from face_destyle.data.metadata import write_jsonl
from face_destyle.schemas import ImageRecord

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def stable_id(relative_path: Path, style_category: str) -> str:
    normalized = relative_path.as_posix().casefold()
    return hashlib.sha256(f"{style_category}:{normalized}".encode()).hexdigest()[:16]


def scan(input_dir: Path, style_category: str) -> list[ImageRecord]:
    files = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    return [
        ImageRecord(
            id=stable_id(path.relative_to(input_dir), style_category),
            image_path=path,
            style_category=style_category,
            source_id=stable_id(path.relative_to(input_dir), style_category),
        )
        for path in files
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--style-category", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    records = scan(args.input_dir, args.style_category)
    if args.dry_run:
        print(f"Would write {len(records)} records to {args.output}")
    else:
        write_jsonl(records, args.output)
        print(f"Wrote {len(records)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
