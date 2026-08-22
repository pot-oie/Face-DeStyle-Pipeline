#!/usr/bin/env python3
"""Machine-validate the frozen Clay/Needle-felt extension pilot inputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

from face_destyle.data.manifests import load_dataset_manifest
from face_destyle.data.metadata import read_jsonl
from face_destyle.schemas import DatasetManifestRecord
from face_destyle.utils.io import load_yaml

EXPECTED_COUNTS = {"clay": 5, "needle_felt": 5}


def validate_material_extension(
    *,
    data_root: Path,
    manifest: Path,
    provenance: Path,
    formal_manifest: Path,
    styles_config: Path,
) -> dict[str, object]:
    selected = load_dataset_manifest(manifest, data_root=data_root, split="extension")
    declared = read_jsonl(manifest, DatasetManifestRecord)
    provenance_rows = [
        json.loads(line)
        for line in provenance.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    provenance_by_id = {str(row["source_id"]): row for row in provenance_rows}
    if len(provenance_by_id) != len(provenance_rows):
        raise ValueError("extension provenance contains duplicate source_id")

    style_counts = Counter(record.style_category for record in selected)
    if style_counts != Counter(EXPECTED_COUNTS):
        raise ValueError(f"extension style counts differ: {dict(style_counts)}")
    if len({record.source_id for record in selected}) != 10:
        raise ValueError("extension pilot must contain ten unique source IDs")
    if len({record.sha256 for record in declared}) != 10:
        raise ValueError("extension pilot must contain ten unique file hashes")

    image_rows = []
    for record in selected:
        source = Path(record.image_path)
        with Image.open(source) as image:
            decoded = ImageOps.exif_transpose(image).convert("RGB")
            decoded.load()
            width, height = decoded.size
        if min(width, height) < 1024:
            raise ValueError(f"{record.source_id}: shortest image side is below 1024")
        private = provenance_by_id.get(record.source_id)
        if private is None or private.get("qc_status") != "accepted":
            raise ValueError(f"{record.source_id}: missing accepted provenance")
        if private.get("style_category") != record.style_category:
            raise ValueError(f"{record.source_id}: provenance style mismatch")
        image_rows.append(
            {
                "source_id": record.source_id,
                "style_category": record.style_category,
                "mode_after_decode": "RGB",
                "width": width,
                "height": height,
            }
        )

    formal = read_jsonl(formal_manifest, DatasetManifestRecord)
    selected_ids = {record.source_id for record in declared}
    selected_groups = {record.source_group_id for record in declared}
    selected_hashes = {record.sha256 for record in declared}
    if selected_ids & {record.source_id for record in formal}:
        raise ValueError("extension source_id overlaps formal-v1")
    if selected_groups & {record.source_group_id for record in formal}:
        raise ValueError("extension source group overlaps formal-v1")
    if selected_hashes & {record.sha256 for record in formal}:
        raise ValueError("extension file hash overlaps formal-v1")

    styles = load_yaml(styles_config).get("styles", {})
    for style in EXPECTED_COUNTS:
        config = styles.get(style, {})
        if not config.get("stage1_prompt") or not config.get("stage2_prompt"):
            raise ValueError(f"{style}: stage1/stage2 prompts are incomplete")

    return {
        "schema": "face-destyle-material-extension-input-validation/v1",
        "status": "passed",
        "candidate_count": len(selected),
        "style_counts": dict(sorted(style_counts.items())),
        "unique_source_ids": len(selected_ids),
        "unique_source_groups": len(selected_groups),
        "unique_file_hashes": len(selected_hashes),
        "minimum_short_side": min(min(row["width"], row["height"]) for row in image_rows),
        "formal_v1_overlap": {"source_id": 0, "source_group_id": 0, "file_hash": 0},
        "images": image_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--styles-config", type=Path, default=Path("configs/styles.yaml"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = validate_material_extension(
            data_root=args.data_root,
            manifest=args.manifest,
            provenance=args.provenance,
            formal_manifest=args.formal_manifest,
            styles_config=args.styles_config,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
