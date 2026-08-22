import importlib.util
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_material_extension.py"
SPEC = importlib.util.spec_from_file_location("validate_material_extension", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def build_fixture(tmp_path: Path) -> dict[str, Path]:
    data_root = tmp_path / "data"
    manifest_rows = []
    provenance_rows = []
    for style in ("clay", "needle_felt"):
        for index in range(5):
            source_id = f"{style}-{index}"
            relative = Path("extension") / style / f"{source_id}.png"
            path = data_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (1024 + index, 1100), style == "clay" and "brown" or "gray").save(
                path
            )
            import hashlib

            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_rows.append(
                {
                    "id": source_id,
                    "source_id": source_id,
                    "source_group_id": source_id,
                    "asset_path": str(relative),
                    "style_category": style,
                    "split": "extension",
                    "sha256": digest,
                    "qc_status": "accepted",
                }
            )
            provenance_rows.append(
                {"source_id": source_id, "style_category": style, "qc_status": "accepted"}
            )
    manifest = tmp_path / "extension.jsonl"
    provenance = tmp_path / "provenance.jsonl"
    formal = tmp_path / "formal.jsonl"
    styles = tmp_path / "styles.yaml"
    write_jsonl(manifest, manifest_rows)
    write_jsonl(provenance, provenance_rows)
    write_jsonl(
        formal,
        [
            {
                "id": "formal-a",
                "source_id": "formal-a",
                "source_group_id": "formal-a",
                "asset_path": "formal-a.png",
                "style_category": "3d_cartoon",
                "split": "pilot",
                "sha256": "f" * 64,
                "qc_status": "accepted",
            }
        ],
    )
    styles.write_text(
        yaml.safe_dump(
            {
                "styles": {
                    style: {"stage1_prompt": "stage one", "stage2_prompt": "stage two"}
                    for style in ("clay", "needle_felt")
                }
            }
        ),
        encoding="utf-8",
    )
    return {
        "data_root": data_root,
        "manifest": manifest,
        "provenance": provenance,
        "formal_manifest": formal,
        "styles_config": styles,
    }


def test_material_extension_validation_passes_for_balanced_pilot(tmp_path: Path) -> None:
    result = MODULE.validate_material_extension(**build_fixture(tmp_path))

    assert result["status"] == "passed"
    assert result["candidate_count"] == 10
    assert result["style_counts"] == {"clay": 5, "needle_felt": 5}
    assert result["minimum_short_side"] == 1024


def test_material_extension_validation_rejects_formal_overlap(tmp_path: Path) -> None:
    paths = build_fixture(tmp_path)
    rows = [json.loads(line) for line in paths["formal_manifest"].read_text().splitlines()]
    rows[0]["source_id"] = "clay-0"
    write_jsonl(paths["formal_manifest"], rows)

    with pytest.raises(ValueError, match="overlaps formal-v1"):
        MODULE.validate_material_extension(**paths)
