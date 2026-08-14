import hashlib
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from face_destyle.data.metadata import write_jsonl
from face_destyle.schemas import DestylizationRecord, ImageRecord

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "run_destylization.py"


def test_single_image_copy_cli(tmp_path):
    source = tmp_path / "single.png"
    Image.new("RGB", (16, 16), (1, 2, 3)).save(source)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(source),
            "--style-category",
            "comic",
            "--record-id",
            "single",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--backend",
            "copy",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (tmp_path / "outputs" / "single.png").exists()


def test_jsonl_batch_copy_cli(tmp_path):
    source = tmp_path / "batch.png"
    Image.new("RGB", (16, 16), (4, 5, 6)).save(source)
    metadata = tmp_path / "metadata.jsonl"
    records_output = tmp_path / "records.jsonl"
    write_jsonl(
        [ImageRecord(id="batch", source_id="batch", image_path=source, style_category="ink")],
        metadata,
    )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--metadata",
            str(metadata),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--records-output",
            str(records_output),
            "--backend",
            "copy",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    rows = records_output.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert DestylizationRecord.model_validate_json(rows[0]).backend == "copy"


def test_portable_manifest_copy_cli(tmp_path):
    data_root = tmp_path / "dataset"
    source = data_root / "raw/comic/portable.png"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (16, 16), (7, 8, 9)).save(source)
    manifest = tmp_path / "manifest.jsonl"
    row = {
        "id": "portable",
        "source_id": "portable-source",
        "source_group_id": "portable-group",
        "asset_path": "raw/comic/portable.png",
        "style_category": "comic",
        "split": "pilot",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "qc_status": "accepted",
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    records_output = tmp_path / "records.jsonl"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--data-root",
            str(data_root),
            "--split",
            "pilot",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--records-output",
            str(records_output),
            "--backend",
            "copy",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    record = DestylizationRecord.model_validate_json(records_output.read_text(encoding="utf-8"))
    assert record.source_id == "portable-source"
    assert record.input_path == source
