"""Resolve frozen dataset manifests against a separate, private data root."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from face_destyle.data.metadata import read_jsonl
from face_destyle.schemas import DatasetManifestRecord, ImageRecord


def resolve_data_root(data_root: str | Path | None = None) -> Path:
    value = data_root if data_root is not None else os.environ.get("FACE_DESTYLE_DATA_ROOT")
    if not value:
        raise RuntimeError("--data-root or FACE_DESTYLE_DATA_ROOT is required with --manifest")
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data root does not exist or is not a directory: {root}")
    return root


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset_manifest(
    manifest_path: str | Path,
    *,
    data_root: str | Path | None = None,
    split: str | None = None,
) -> list[ImageRecord]:
    """Validate a frozen manifest and return runtime records with absolute image paths."""
    records = read_jsonl(manifest_path, DatasetManifestRecord)
    root = resolve_data_root(data_root)
    source_ids: set[str] = set()
    group_splits: dict[str, str] = {}
    resolved: list[ImageRecord] = []

    for record in records:
        if record.source_id in source_ids:
            raise ValueError(f"duplicate source_id in manifest: {record.source_id}")
        source_ids.add(record.source_id)
        previous_split = group_splits.setdefault(record.source_group_id, record.split)
        if previous_split != record.split:
            raise ValueError(
                f"source_group_id {record.source_group_id!r} crosses splits: "
                f"{previous_split} and {record.split}"
            )

        candidate = (root / record.asset_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"asset_path escapes the data root: {record.asset_path}") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"missing manifest asset for {record.id}: {candidate}")
        actual_sha256 = file_sha256(candidate)
        if actual_sha256 != record.sha256:
            raise ValueError(
                f"checksum mismatch for {record.id}: expected {record.sha256}, "
                f"got {actual_sha256}"
            )
        if split is None or record.split == split:
            resolved.append(
                ImageRecord(
                    id=record.id,
                    source_id=record.source_id,
                    image_path=candidate,
                    style_category=record.style_category,
                )
            )

    if split is not None and not resolved:
        raise ValueError(f"manifest contains no records for split {split!r}")
    return resolved
