import hashlib
import importlib.util
import zipfile
from pathlib import Path

from PIL import Image

from face_destyle.data.metadata import write_jsonl
from face_destyle.filtering.prompt_rewriter import select_prompt
from face_destyle.schemas import DatasetManifestRecord, DestylizationRecord
from face_destyle.utils.io import load_yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_flux_test_archive.py"
SPEC = importlib.util.spec_from_file_location("validate_flux_test_archive", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_machine_validates_complete_frozen_flux_archive(tmp_path: Path) -> None:
    run = tmp_path / "formal-v1-test-flux-native1024-seed42"
    images = run / "images"
    images.mkdir(parents=True)
    styles = ("3d_cartoon", "comic", "ink", "watercolor")
    styles_config = ROOT / "configs/styles.yaml"
    prompt_config = load_yaml(styles_config)
    manifest_rows = []
    generation_rows = []
    for style_index, style in enumerate(styles):
        for index in range(15):
            source_id = f"{style}-{index:02d}"
            asset = Path("raw") / style / f"{source_id}.png"
            manifest_rows.append(
                DatasetManifestRecord(
                    id=source_id,
                    source_id=source_id,
                    source_group_id=f"group-{source_id}",
                    asset_path=asset,
                    style_category=style,
                    split="test",
                    sha256="0" * 64,
                )
            )
            output = images / f"{source_id}.png"
            Image.new("RGB", (1024, 1024), (style_index, index, 1)).save(output)
            generation_rows.append(
                DestylizationRecord(
                    id=source_id,
                    source_id=source_id,
                    input_path=Path("/server/data") / asset,
                    output_path=Path("/server/run/images") / output.name,
                    style_category=style,
                    backend=MODULE.BACKEND,
                    seed=42,
                    prompt=select_prompt(style, prompt_config, adaptive=True),
                    extra={
                        "official_model_id": "black-forest-labs/FLUX.1-Kontext-dev",
                        "source_revision": "master",
                        "dtype": "bfloat16",
                        "batch_size": 1,
                        "height": 1024,
                        "width": 1024,
                        "max_area": 1024 * 1024,
                        "output_height": 1024,
                        "output_width": 1024,
                        "guidance_scale": 2.5,
                        "num_inference_steps": 28,
                        "offload": "enable_model_cpu_offload",
                        "local_files_only": True,
                        "transport_source": "modelscope_mirror",
                        "transport_model_id": "black-forest-labs/FLUX.1-Kontext-dev",
                        "resolved_model_path": "/server/model/FLUX.1-Kontext-dev",
                        "download_manifest_sha256": "1" * 64,
                        "hash_manifest_sha256": "2" * 64,
                        "package_versions": {"diffusers": "test"},
                    },
                )
            )
    manifest = tmp_path / "inputs.jsonl"
    write_jsonl(manifest_rows, manifest)
    write_jsonl(generation_rows, run / "records.jsonl")
    (run / "failures.jsonl").write_text("", encoding="utf-8")
    archive = tmp_path / f"{run.name}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(run.rglob("*")):
            if path.is_file():
                bundle.write(path, arcname=f"{run.name}/{path.relative_to(run)}")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    sidecar = archive.with_name(f"{archive.name}.sha256")
    sidecar.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")

    report = MODULE.validate_archive(archive, sidecar, manifest, styles_config)

    assert report["success_record_count"] == 60
    assert report["unique_test_source_count"] == 60
    assert report["image_count"] == 60
    assert report["failure_record_count"] == 0
    assert report["zip_crc"] == "passed"
