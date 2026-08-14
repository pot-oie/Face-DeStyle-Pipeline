import hashlib
import json

import pytest
from pydantic import ValidationError

from face_destyle.data.manifests import load_dataset_manifest
from face_destyle.schemas import DatasetManifestRecord


def manifest_row(path, payload: bytes, **overrides):
    row = {
        "id": "sample",
        "source_id": "source",
        "source_group_id": "group",
        "asset_path": path,
        "style_category": "comic",
        "split": "pilot",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "qc_status": "accepted",
    }
    row.update(overrides)
    return row


def write_manifest(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_manifest_resolves_relative_assets_and_filters_split(tmp_path):
    data_root = tmp_path / "data"
    image = data_root / "raw/comic/sample.png"
    image.parent.mkdir(parents=True)
    payload = b"image-bytes"
    image.write_bytes(payload)
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [manifest_row("raw/comic/sample.png", payload)])

    records = load_dataset_manifest(manifest, data_root=data_root, split="pilot")

    assert len(records) == 1
    assert records[0].image_path == image
    assert records[0].source_id == "source"


def test_manifest_uses_data_root_environment(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    image = data_root / "raw/ink/sample.png"
    image.parent.mkdir(parents=True)
    payload = b"ink-image"
    image.write_bytes(payload)
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        manifest,
        [manifest_row("raw/ink/sample.png", payload, style_category="ink")],
    )
    monkeypatch.setenv("FACE_DESTYLE_DATA_ROOT", str(data_root))

    assert load_dataset_manifest(manifest)[0].image_path == image


def test_manifest_rejects_checksum_mismatch(tmp_path):
    data_root = tmp_path / "data"
    image = data_root / "raw/comic/sample.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"actual")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [manifest_row("raw/comic/sample.png", b"expected")])

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_dataset_manifest(manifest, data_root=data_root)


def test_manifest_rejects_source_group_split_leakage(tmp_path):
    data_root = tmp_path / "data"
    rows = []
    for index, split in enumerate(("pilot", "test")):
        payload = f"image-{index}".encode()
        relative = f"raw/comic/{index}.png"
        image = data_root / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(payload)
        rows.append(
            manifest_row(
                relative,
                payload,
                id=f"sample-{index}",
                source_id=f"source-{index}",
                source_group_id="shared-group",
                split=split,
            )
        )
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, rows)

    with pytest.raises(ValueError, match="crosses splits"):
        load_dataset_manifest(manifest, data_root=data_root)


def test_manifest_schema_rejects_absolute_or_parent_paths():
    payload = b"image"
    for path in ("/private/image.png", "raw/../private/image.png"):
        with pytest.raises(ValidationError, match="relative to the data root"):
            DatasetManifestRecord.model_validate(manifest_row(path, payload))
