import hashlib
import importlib.util
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_workspace_assets.py"
SPEC = importlib.util.spec_from_file_location("audit_workspace_assets", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_formal_manifest_inventory_and_run_scan(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    image = data_root / "raw" / "comic" / "source.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (16, 16), "red").save(image)
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    manifest = tmp_path / "inputs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "source",
                "source_id": "source",
                "source_group_id": "group",
                "asset_path": "raw/comic/source.png",
                "style_category": "comic",
                "split": "calibration",
                "sha256": digest,
                "qc_status": "accepted",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    inventory, source_splits = MODULE.formal_manifest_inventory(
        manifest, data_root, verify_checksums=True
    )

    assert inventory["records"] == 1
    assert inventory["checksum_validation"] == "passed"
    assert source_splits == {"source": "calibration"}

    workspace = tmp_path / "workspace"
    run_dir = workspace / "outputs" / "run"
    output = run_dir / "images" / "source.png"
    output.parent.mkdir(parents=True)
    Image.new("RGB", (16, 16), "blue").save(output)
    records = run_dir / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "source_id": "source",
                "output_path": str(output),
                "backend": "diffusers",
                "seed": 42,
                "style_category": "comic",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    run = MODULE.scan_run(records, workspace, source_splits)

    assert run["records"] == 1
    assert run["recorded_outputs_present"] == 1
    assert run["formal_split_coverage"] == {"calibration": 1}
