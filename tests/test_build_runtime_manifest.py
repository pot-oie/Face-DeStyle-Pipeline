import hashlib
import json
from pathlib import Path

from PIL import Image

from face_destyle.data.runtime_manifests import build_runtime_manifest


def rich_row(path: Path, payload: bytes, source_id: str, style: str) -> dict[str, object]:
    return {
        "id": source_id,
        "source_id": source_id,
        "source_group_id": source_id,
        "local_path": str(path),
        "style_category": style,
        "split": "pilot",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "qc_status": "accepted",
        "provider": "private-pilot",
    }


def test_build_runtime_manifest_validates_and_limits_by_style(tmp_path):
    rows = []
    for style, count in (("comic", 2), ("ink", 1)):
        for index in range(count):
            path = tmp_path / "raw" / style / f"{style}-{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (index, 2, 3)).save(path)
            rows.append(rich_row(path, path.read_bytes(), path.stem, style))
    source = tmp_path / "pilot.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in reversed(rows)) + "\n")

    records = build_runtime_manifest(
        source,
        tmp_path,
        split="pilot",
        path_anchor="raw",
        limit_per_style=1,
    )

    assert [(record.style_category, record.source_id) for record in records] == [
        ("comic", "comic-0"),
        ("ink", "ink-0"),
    ]
    assert records[0].asset_path == Path("raw/comic/comic-0.png")
