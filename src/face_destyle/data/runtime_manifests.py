"""Convert rich private provenance rows into strict portable runtime manifests."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from face_destyle.data.manifests import file_sha256
from face_destyle.schemas import DatasetManifestRecord


def relative_asset_path(row: dict[str, Any], path_anchor: str) -> Path:
    if row.get("asset_path"):
        return Path(str(row["asset_path"]))
    local_path = Path(str(row.get("local_path", "")))
    if not local_path.parts:
        raise ValueError("record has neither asset_path nor local_path")
    try:
        anchor_index = local_path.parts.index(path_anchor)
    except ValueError as exc:
        raise ValueError(
            f"local_path does not contain path anchor {path_anchor!r}: {local_path}"
        ) from exc
    return Path(*local_path.parts[anchor_index:])


def build_runtime_manifest(
    source: Path,
    data_root: Path,
    *,
    split: str,
    path_anchor: str,
    limit_per_style: int | None,
) -> list[DatasetManifestRecord]:
    candidates: list[DatasetManifestRecord] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("qc_status") != "accepted" or row.get("split") != split:
            continue
        asset_path = relative_asset_path(row, path_anchor)
        image_path = data_root / asset_path
        if not image_path.is_file():
            raise FileNotFoundError(f"line {line_number}: missing image: {image_path}")
        expected_sha256 = str(row.get("sha256", ""))
        actual_sha256 = file_sha256(image_path)
        if expected_sha256 != actual_sha256:
            raise ValueError(
                f"line {line_number}: checksum mismatch for {image_path}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        source_id = str(row["source_id"])
        candidates.append(
            DatasetManifestRecord(
                id=str(row.get("id") or source_id),
                source_id=source_id,
                source_group_id=str(row["source_group_id"]),
                asset_path=asset_path,
                style_category=str(row["style_category"]),
                split=split,
                sha256=actual_sha256,
                qc_status="accepted",
            )
        )

    candidates.sort(key=lambda record: (record.style_category, record.source_id))
    if limit_per_style is None:
        return candidates
    selected: list[DatasetManifestRecord] = []
    counts: defaultdict[str, int] = defaultdict(int)
    for record in candidates:
        if counts[record.style_category] < limit_per_style:
            selected.append(record)
            counts[record.style_category] += 1
    return selected
