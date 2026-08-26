#!/usr/bin/env python3
"""Build the frozen non-Origami multistyle routing-validation manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from face_destyle.data.manifests import file_sha256
from face_destyle.data.metadata import read_jsonl
from face_destyle.schemas import DatasetManifestRecord

STYLE_ORDER = ("comic", "ink", "watercolor", "3d_cartoon", "clay", "needle_felt")
EXPECTED_COUNTS = {
    "comic": 24,
    "ink": 24,
    "watercolor": 24,
    "3d_cartoon": 24,
    "clay": 24,
    "needle_felt": 17,
}


def portable_record(
    *, source_id: str, asset_path: Path, style: str, data_root: Path
) -> DatasetManifestRecord:
    image_path = data_root / asset_path
    if not image_path.is_file():
        raise FileNotFoundError(f"missing selected image: {image_path}")
    return DatasetManifestRecord(
        id=source_id,
        source_id=source_id,
        source_group_id=source_id,
        asset_path=asset_path,
        style_category=style,
        split="extension",
        sha256=file_sha256(image_path),
        qc_status="accepted",
    )


def load_pair_bank(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_records(repo: Path, data_root: Path) -> list[DatasetManifestRecord]:
    gap_ids = {
        record.source_id
        for record in read_jsonl(
            repo / "data/manifests/multistyle-routing/missing_stage12_sources.jsonl",
            DatasetManifestRecord,
        )
    }
    formal = read_jsonl(
        repo / "data/manifests/formal-v1/inputs.jsonl", DatasetManifestRecord
    )
    records: list[DatasetManifestRecord] = []
    for record in formal:
        if record.style_category in {"comic", "ink", "watercolor"}:
            if record.source_id not in gap_ids:
                records.append(
                    portable_record(
                        source_id=record.source_id,
                        asset_path=record.asset_path,
                        style=record.style_category,
                        data_root=data_root,
                    )
                )

    three_d_rows = load_pair_bank(
        repo / "data/manifests/multistyle-pair-bank/3d_cartoon_sources.csv"
    )
    selected_3d = [
        row
        for row in three_d_rows
        if row["role"] in {"candidate", "holdout"}
        and row["source_id"].startswith("synthetic-3d-cartoon-")
        and int(row["source_id"].rsplit("-", 1)[1]) <= 24
    ]
    for row in selected_3d:
        records.append(
            portable_record(
                source_id=row["source_id"],
                asset_path=Path(row["asset_path"]),
                style="3d_cartoon",
                data_root=data_root,
            )
        )

    clay_rows = load_pair_bank(
        repo / "data/manifests/multistyle-pair-bank/clay_sources.csv"
    )
    for row in clay_rows:
        if row["role"] in {"candidate", "holdout"}:
            records.append(
                portable_record(
                    source_id=row["source_id"],
                    asset_path=Path(row["asset_path"]),
                    style="clay",
                    data_root=data_root,
                )
            )

    needle_v1 = read_jsonl(
        data_root
        / "extensions/material_styles_v1/manifests/extension_pilot_inputs.jsonl",
        DatasetManifestRecord,
    )
    for record in needle_v1:
        if record.style_category == "needle_felt":
            records.append(
                portable_record(
                    source_id=record.source_id,
                    asset_path=record.asset_path,
                    style="needle_felt",
                    data_root=data_root,
                )
            )

    v2_records_path = data_root / "extensions/material_styles_v2/generation_records.jsonl"
    with v2_records_path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("style") != "needle_felt":
                continue
            if payload.get("qc_status") != "kept_after_simple_visual_review":
                continue
            source_id = Path(payload["filename"]).stem
            records.append(
                portable_record(
                    source_id=source_id,
                    asset_path=Path(
                        f"extensions/material_styles_v2/raw/needle_felt/{payload['filename']}"
                    ),
                    style="needle_felt",
                    data_root=data_root,
                )
            )

    style_rank = {style: index for index, style in enumerate(STYLE_ORDER)}
    records.sort(key=lambda record: (style_rank[record.style_category], record.source_id))
    source_ids = [record.source_id for record in records]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("selected validation records contain duplicate source IDs")
    counts = {
        style: sum(record.style_category == style for record in records)
        for style in STYLE_ORDER
    }
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"unexpected validation counts: {counts}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/manifests/multistyle-routing/non_origami_validation_137.jsonl"
        ),
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    data_root = args.data_root.expanduser().resolve()
    output = args.output if args.output.is_absolute() else repo / args.output
    if output.exists():
        parser.error(f"refusing to overwrite existing manifest: {output}")
    try:
        records = build_records(repo, data_root)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
    print(f"Wrote {len(records)} records to {output}")
    print("Counts: " + ", ".join(f"{key}={value}" for key, value in EXPECTED_COUNTS.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
