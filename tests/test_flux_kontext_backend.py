from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from face_destyle.pipelines.flux_kontext_backend import FluxKontextBackend, FluxKontextSettings
from face_destyle.schemas import ImageRecord


class FakePipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=[Image.new("RGB", (1024, 1024), (12, 34, 56))])


def make_settings(tmp_path: Path) -> FluxKontextSettings:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
    for component in (
        "scheduler",
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "transformer",
        "vae",
    ):
        (model_dir / component).mkdir()
    download_manifest = tmp_path / "download.txt"
    hash_manifest = tmp_path / "hashes.sha256"
    download_manifest.write_text("source=modelscope_mirror\n", encoding="utf-8")
    hash_manifest.write_text("test hashes\n", encoding="utf-8")
    return FluxKontextSettings(model_dir, download_manifest, hash_manifest)


def test_mock_kontext_probe_records_frozen_settings(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (900, 700)).save(source)
    fake = FakePipeline()
    backend = FluxKontextBackend(
        make_settings(tmp_path),
        {"styles": {"comic": {"stage1_prompt": "make a natural photograph"}}},
        pipeline_factory=lambda _path: fake,
    )
    record = ImageRecord(id="sample", source_id="source", image_path=source, style_category="comic")

    result = backend.run(record, tmp_path / "outputs", seed=42)

    assert result.backend == "flux1_kontext_dev_prompt_edit_bf16_offloaded"
    assert result.output_path.is_file()
    assert result.extra["dtype"] == "bfloat16"
    assert result.extra["offload"] == "enable_model_cpu_offload"
    assert result.extra["transport_source"] == "modelscope_mirror"
    assert fake.calls[0]["image"].size == (1024, 1024)
    assert fake.calls[0]["height"] == 1024
    assert fake.calls[0]["width"] == 1024
    assert fake.calls[0]["max_area"] == 1024 * 1024
    assert fake.calls[0]["guidance_scale"] == 2.5
    assert fake.calls[0]["num_inference_steps"] == 28
    assert fake.calls[0]["generator"] == 42
    assert result.extra["output_height"] == 1024
    assert result.extra["output_width"] == 1024


def test_mock_kontext_probe_uses_declared_stage2_prompt(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (700, 900)).save(source)
    fake = FakePipeline()
    backend = FluxKontextBackend(
        make_settings(tmp_path),
        {
            "styles": {
                "clay": {
                    "stage1_prompt": "generic clay conversion",
                    "stage2_prompt": "remove persistent terracotta material",
                }
            }
        },
        prompt_stage="stage2",
        pipeline_factory=lambda _path: fake,
    )

    result = backend.run(
        ImageRecord(id="clay-a", source_id="clay-a", image_path=source, style_category="clay"),
        tmp_path / "outputs",
        seed=42,
    )

    assert result.prompt == "remove persistent terracotta material"
    assert result.extra["prompt_stage"] == "stage2"
    assert fake.calls[0]["prompt"] == "remove persistent terracotta material"


def test_mock_kontext_probe_uses_source_specific_prompt_override(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (800, 800)).save(source)
    fake = FakePipeline()
    backend = FluxKontextBackend(
        make_settings(tmp_path),
        {"styles": {"origami": {"stage1_prompt": "generic origami prompt"}}},
        prompt_overrides={"origami-hard": "naturalize hair and clothing"},
        pipeline_factory=lambda _path: fake,
    )

    result = backend.run(
        ImageRecord(
            id="origami-hard",
            source_id="origami-hard",
            image_path=source,
            style_category="origami",
        ),
        tmp_path / "outputs",
        seed=42,
    )

    assert result.prompt == "naturalize hair and clothing"
    assert result.extra["prompt_override"] is True
    assert fake.calls[0]["prompt"] == "naturalize hair and clothing"
