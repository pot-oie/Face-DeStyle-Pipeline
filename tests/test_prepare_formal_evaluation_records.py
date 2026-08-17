import hashlib
import json
import runpy
import sys
from pathlib import Path

from PIL import Image

from face_destyle.data.metadata import read_jsonl, write_jsonl
from face_destyle.schemas import DatasetManifestRecord, DestylizationRecord


def test_prepare_formal_evaluation_records_selects_only_requested_split(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    generated = tmp_path / "generated"
    manifest_rows = []
    generation_rows = []
    for split, color in (("calibration", "red"), ("test", "blue")):
        source = data_root / "raw" / f"{split}.png"
        output = generated / f"{split}.png"
        source.parent.mkdir(parents=True, exist_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), color).save(source)
        Image.new("RGB", (16, 16), color).save(output)
        manifest_rows.append(
            DatasetManifestRecord(
                id=split,
                source_id=split,
                source_group_id=split,
                asset_path=source.relative_to(data_root),
                style_category="comic",
                split=split,
                sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            )
        )
        generation_rows.append(
            DestylizationRecord(
                id=split,
                source_id=split,
                input_path=source,
                output_path=output,
                style_category="comic",
                backend="diffusers",
                seed=42,
            )
        )

    manifest = tmp_path / "manifest.jsonl"
    records = tmp_path / "records.jsonl"
    output_dir = tmp_path / "selected"
    write_jsonl(manifest_rows, manifest)
    write_jsonl(generation_rows, records)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_formal_evaluation_records.py",
            "--manifest",
            str(manifest),
            "--data-root",
            str(data_root),
            "--split",
            "calibration",
            "--records",
            f"method-a={records}",
            "--output-dir",
            str(output_dir),
        ],
    )

    try:
        runpy.run_path("scripts/prepare_formal_evaluation_records.py", run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0

    selected = read_jsonl(output_dir / "method-a.records.jsonl", DestylizationRecord)
    assert [record.source_id for record in selected] == ["calibration"]
    summary = json.loads((output_dir / "selection.json").read_text(encoding="utf-8"))
    assert summary["expected_records_per_method"] == 1
    assert summary["methods"][0]["source_record_count"] == 2
