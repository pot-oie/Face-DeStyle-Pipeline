#!/usr/bin/env python3
"""Select and validate one frozen split from complete generation records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from face_destyle.data.manifests import load_dataset_manifest
from face_destyle.data.metadata import read_jsonl, write_jsonl
from face_destyle.schemas import DestylizationRecord


def parse_records_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("records must use METHOD=PATH")
    method, raw_path = value.split("=", 1)
    if not method.strip() or not raw_path.strip():
        raise ValueError("records must use METHOD=PATH")
    return method.strip(), Path(raw_path).expanduser().resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("pilot", "calibration", "test", "extension"),
        required=True,
    )
    parser.add_argument(
        "--records",
        action="append",
        required=True,
        metavar="METHOD=PATH",
        help="Generation method and complete DestylizationRecord JSONL; repeat per method.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"refusing non-empty output directory: {args.output_dir}")
    methods = [parse_records_argument(value) for value in args.records]
    names = [name for name, _path in methods]
    if len(names) != len(set(names)):
        parser.error("method names must be unique")

    expected_records = load_dataset_manifest(
        args.manifest,
        data_root=args.data_root,
        split=args.split,
    )
    expected_ids = [record.source_id for record in expected_records]
    expected = {record.source_id: record for record in expected_records}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for method, records_path in methods:
        records = read_jsonl(
            records_path,
            DestylizationRecord,
            check_paths=True,
            path_fields=("input_path", "output_path"),
        )
        selected_by_id = {
            record.source_id: record for record in records if record.source_id in expected
        }
        if len(selected_by_id) != sum(record.source_id in expected for record in records):
            parser.error(f"duplicate selected source ID in {method}")
        missing = sorted(set(expected) - set(selected_by_id))
        if missing:
            parser.error(f"{method} lacks {len(missing)} {args.split} records; first={missing[0]}")
        selected = [selected_by_id[source_id] for source_id in expected_ids]
        for record in selected:
            actual_input = Path(record.input_path).resolve()
            manifest_record = expected[record.source_id]
            expected_input = Path(manifest_record.image_path).resolve()
            if actual_input != expected_input:
                parser.error(
                    f"{method}/{record.source_id} input mismatch: "
                    f"{actual_input} != {expected_input}"
                )
            if record.style_category != manifest_record.style_category:
                parser.error(
                    f"{method}/{record.source_id} style mismatch: "
                    f"{record.style_category} != {manifest_record.style_category}"
                )

        output_path = args.output_dir / f"{method}.records.jsonl"
        write_jsonl(selected, output_path)
        summaries.append(
            {
                "method": method,
                "source_records": str(records_path),
                "source_record_count": len(records),
                "selected_split": args.split,
                "selected_record_count": len(selected),
                "selected_records": str(output_path.resolve()),
                "selected_records_sha256": file_sha256(output_path),
                "backends": sorted({record.backend for record in selected}),
                "seeds": sorted({record.seed for record in selected}),
            }
        )

    summary_path = args.output_dir / "selection.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema": "face-destyle-evaluation-record-selection/v1",
                "manifest": str(args.manifest.resolve()),
                "split": args.split,
                "expected_records_per_method": len(expected_records),
                "methods": summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Prepared {len(methods)} methods x {len(expected_records)} {args.split} records "
        f"in {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
