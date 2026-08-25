import importlib.util
import json
from pathlib import Path

from PIL import Image

from face_destyle.schemas import ImageRecord

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_flux_kontext_probe.py"
SPEC = importlib.util.spec_from_file_location("run_flux_kontext_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def record(source_id: str, style: str) -> ImageRecord:
    return ImageRecord(
        id=source_id,
        source_id=source_id,
        image_path=Path(f"/{source_id}.png"),
        style_category=style,
    )


def test_probe_selection_is_fixed_by_style_and_source_id():
    records = [
        record("comic-z", "comic"),
        record("watercolor-a", "watercolor"),
        record("ink-a", "ink"),
        record("3d-b", "3d_cartoon"),
        record("comic-a", "comic"),
        record("3d-a", "3d_cartoon"),
    ]

    assert [item.source_id for item in MODULE.select_probe_records(records, "all")] == [
        "3d-a",
        "comic-a",
        "ink-a",
        "watercolor-a",
    ]
    assert [item.source_id for item in MODULE.select_probe_records(records, "first")] == [
        "3d-a"
    ]
    assert [item.source_id for item in MODULE.select_probe_records(records, "remaining")] == [
        "comic-a",
        "ink-a",
        "watercolor-a",
    ]
    assert [item.source_id for item in MODULE.select_probe_records(records, "pilot")] == [
        "3d-a",
        "3d-b",
        "comic-a",
        "comic-z",
        "ink-a",
        "watercolor-a",
    ]
    assert MODULE.select_probe_records(records, "batch") == MODULE.select_probe_records(
        records, "pilot"
    )


def test_resume_validates_success_record_and_output(tmp_path):
    source = tmp_path / "source.png"
    output_dir = tmp_path / "images"
    output = output_dir / "source-a.png"
    output_dir.mkdir()
    Image.new("RGB", (8, 8), "red").save(source)
    Image.new("RGB", (1024, 1024), "blue").save(output)
    selected = [
        ImageRecord(
            id="source-a",
            source_id="source-a",
            image_path=source,
            style_category="comic",
        )
    ]
    settings = MODULE.FluxKontextSettings(
        model_dir=tmp_path / "model",
        download_manifest=tmp_path / "download.txt",
        hash_manifest=tmp_path / "hashes.txt",
    )
    styles = {"styles": {"comic": {"stage1_prompt": "comic prompt"}}}
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "id": "source-a",
                "source_id": "source-a",
                "input_path": str(source),
                "output_path": str(output),
                "style_category": "comic",
                "backend": MODULE.FluxKontextBackend.name,
                "seed": 42,
                "prompt": "comic prompt",
                "extra": {
                    "resolved_model_path": str(settings.model_dir.resolve()),
                    "download_manifest": str(settings.download_manifest.resolve()),
                    "hash_manifest": str(settings.hash_manifest.resolve()),
                    "source_revision": "master",
                    "dtype": "bfloat16",
                    "batch_size": 1,
                    "height": 1024,
                    "width": 1024,
                    "guidance_scale": 2.5,
                    "num_inference_steps": 28,
                    "offload": "enable_model_cpu_offload",
                    "local_files_only": True,
                    "lora_weights": None,
                    "lora_scale": 1.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    completed = MODULE.validate_resume_state(
        records_path=records,
        failures_path=tmp_path / "failures.jsonl",
        output_dir=output_dir,
        selected=selected,
        settings=settings,
        styles_config=styles,
        seed=42,
    )

    assert completed == {"source-a"}


def test_resume_rejects_unexplained_output(tmp_path):
    output_dir = tmp_path / "images"
    output_dir.mkdir()
    Image.new("RGB", (8, 8), "red").save(output_dir / "stray.png")
    settings = MODULE.FluxKontextSettings(
        model_dir=tmp_path / "model",
        download_manifest=tmp_path / "download.txt",
        hash_manifest=tmp_path / "hashes.txt",
    )

    try:
        MODULE.validate_resume_state(
            records_path=tmp_path / "records.jsonl",
            failures_path=tmp_path / "failures.jsonl",
            output_dir=output_dir,
            selected=[record("source-a", "comic")],
            settings=settings,
            styles_config={"styles": {"comic": {"stage1_prompt": "comic prompt"}}},
            seed=42,
        )
    except ValueError as exc:
        assert "unexplained" in str(exc)
    else:
        raise AssertionError("unexplained resume output was accepted")


def test_material_extension_selection_accepts_two_declared_styles():
    records = [
        record("felt-b", "needle_felt"),
        record("clay-a", "clay"),
        record("felt-a", "needle_felt"),
        record("unrelated", "comic"),
    ]

    selected = MODULE.select_probe_records(
        records,
        "batch",
        required_styles=("clay", "needle_felt"),
    )

    assert [item.source_id for item in selected] == ["clay-a", "felt-a", "felt-b"]


def test_explicit_source_subset_retains_input_order():
    records = [
        record("source-c", "clay"),
        record("source-a", "clay"),
        record("source-b", "clay"),
    ]

    selected = MODULE.filter_records_by_source_ids(
        records, ["source-b", "source-c"]
    )

    assert [item.source_id for item in selected] == ["source-c", "source-b"]


def test_explicit_source_subset_rejects_missing_and_duplicate_ids():
    records = [record("source-a", "clay")]

    for requested, message in (
        (["source-a", "source-a"], "unique"),
        (["source-b"], "unavailable"),
    ):
        try:
            MODULE.filter_records_by_source_ids(records, requested)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"invalid source subset was accepted: {requested}")


def test_load_prompt_overrides_requires_non_empty_string_mapping(tmp_path):
    valid = tmp_path / "valid.json"
    valid.write_text(
        json.dumps({"source-a": "  targeted instruction  "}), encoding="utf-8"
    )
    assert MODULE.load_prompt_overrides(valid) == {
        "source-a": "targeted instruction"
    }

    for index, payload in enumerate(({}, [], {"source-a": ""}, {"": "prompt"})):
        invalid = tmp_path / f"invalid-{index}.json"
        invalid.write_text(json.dumps(payload), encoding="utf-8")
        try:
            MODULE.load_prompt_overrides(invalid)
        except ValueError as exc:
            assert "prompt override" in str(exc)
        else:
            raise AssertionError(f"invalid prompt overrides were accepted: {payload}")


def test_load_sequential_inputs_uses_stage1_outputs(tmp_path):
    stage1_output = tmp_path / "stage1" / "source-a.png"
    stage1_output.parent.mkdir()
    Image.new("RGB", (32, 32), "blue").save(stage1_output)
    records = tmp_path / "stage1-records.jsonl"
    records.write_text(
        json.dumps(
            {
                "id": "source-a",
                "source_id": "source-a",
                "input_path": str(tmp_path / "original.png"),
                "output_path": str(stage1_output),
                "style_category": "3d_cartoon",
                "backend": MODULE.FluxKontextBackend.name,
                "seed": 42,
                "prompt": "stage one",
                "extra": {"prompt_stage": "stage1"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = MODULE.load_sequential_inputs(records)

    assert len(loaded) == 1
    assert loaded[0].source_id == "source-a"
    assert loaded[0].image_path == stage1_output.resolve()


def test_load_sequential_inputs_rejects_non_stage1_record(tmp_path):
    output = tmp_path / "output.png"
    Image.new("RGB", (8, 8), "blue").save(output)
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "id": "source-a",
                "source_id": "source-a",
                "input_path": str(tmp_path / "input.png"),
                "output_path": str(output),
                "style_category": "3d_cartoon",
                "backend": MODULE.FluxKontextBackend.name,
                "seed": 42,
                "prompt": "stage two",
                "extra": {"prompt_stage": "stage2"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        MODULE.load_sequential_inputs(records)
    except ValueError as exc:
        assert "not a Stage 1 record" in str(exc)
    else:
        raise AssertionError("non-Stage 1 record was accepted as a sequential input")
