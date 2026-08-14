from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from face_destyle.models import ModelRegistry
from face_destyle.pipelines import CannyControlNetBackend, CannyControlNetSettings
from face_destyle.schemas import ImageRecord


class FakePipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=[Image.new("RGB", (64, 64), (30, 40, 50))])


def make_registry(tmp_path, monkeypatch):
    config = tmp_path / "models.yaml"
    config.write_text(
        """
assets:
  sdxl_base:
    role: generator
    source: huggingface_cache
    model_id: org/sdxl
    revision: base123
    license: test-only
    required_files: [model_index.json]
  canny_controlnet:
    role: structural_control
    source: huggingface_cache
    model_id: org/canny
    revision: canny123
    license: test-only
    required_files: [config.json, diffusion_pytorch_model.fp16.safetensors]
""".strip(),
        encoding="utf-8",
    )
    hub = tmp_path / "hub"
    base = hub / "models--org--sdxl/snapshots/base123"
    control = hub / "models--org--canny/snapshots/canny123"
    base.mkdir(parents=True)
    control.mkdir(parents=True)
    (base / "model_index.json").write_text("{}", encoding="utf-8")
    (control / "config.json").write_text("{}", encoding="utf-8")
    (control / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"weights")
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    return ModelRegistry.from_yaml(config), base, control


def styles_config():
    return {
        "styles": {
            "comic": {
                "stage1_prompt": "realistic portrait",
                "negative_prompt": "comic",
                "canny_low": 25,
                "canny_high": 75,
            }
        }
    }


def test_canny_backend_uses_both_local_snapshots_and_saves_condition(tmp_path, monkeypatch):
    source = tmp_path / "input.png"
    image = Image.new("RGB", (80, 80), "black")
    ImageDraw.Draw(image).rectangle((20, 20, 60, 60), fill="white")
    image.save(source)
    registry, base, control = make_registry(tmp_path, monkeypatch)
    fake = FakePipeline()
    factory_calls = []

    def factory(model_path, control_path, kwargs):
        factory_calls.append((model_path, control_path, kwargs))
        return fake

    backend = CannyControlNetBackend(
        CannyControlNetSettings(
            height=64,
            width=64,
            num_inference_steps=3,
            controlnet_conditioning_scale=0.7,
        ),
        styles_config(),
        registry,
        pipeline_factory=factory,
    )
    record = ImageRecord(
        id="sample",
        source_id="source",
        image_path=source,
        style_category="comic",
    )

    result = backend.run(record, tmp_path / "outputs", seed=42)

    assert factory_calls == [
        (
            str(base),
            str(control),
            {
                "use_safetensors": True,
                "variant": "fp16",
                "local_files_only": True,
            },
        )
    ]
    assert fake.calls[0]["control_image"].size == (64, 64)
    assert fake.calls[0]["controlnet_conditioning_scale"] == 0.7
    assert fake.calls[0]["control_guidance_start"] == 0.0
    assert fake.calls[0]["control_guidance_end"] == 1.0
    assert result.backend == "canny"
    assert result.extra["baseline"] == "global_canny_sdxl_controlnet_img2img"
    assert result.extra["control_model_revision"] == "canny123"
    assert result.extra["resolved_control_model_path"] == str(control)
    assert result.extra["canny_low"] == 25
    assert result.extra["canny_high"] == 75
    assert result.extra["controlnet_conditioning_scale"] == 0.7
    assert (tmp_path / "outputs/sample.canny.png").is_file()


def test_canny_backend_rejects_missing_control_weights(tmp_path, monkeypatch):
    registry, _base, control = make_registry(tmp_path, monkeypatch)
    (control / "diffusion_pytorch_model.fp16.safetensors").unlink()
    source = tmp_path / "input.png"
    Image.new("RGB", (64, 64)).save(source)
    backend = CannyControlNetBackend(
        CannyControlNetSettings(height=64, width=64),
        styles_config(),
        registry,
        pipeline_factory=lambda *_args: FakePipeline(),
    )
    record = ImageRecord(id="sample", source_id="source", image_path=source, style_category="comic")

    with pytest.raises(RuntimeError, match="diffusion_pytorch_model.fp16.safetensors"):
        backend.run(record, tmp_path / "outputs", seed=42)


def test_canny_settings_reject_invalid_control_values():
    for values, message in [
        ({"controlnet_conditioning_scale": 0.0}, "controlnet_conditioning_scale"),
        ({"control_guidance_start": 0.8, "control_guidance_end": 0.2}, "control guidance"),
        ({"canny_low": 200, "canny_high": 100}, "Canny thresholds"),
    ]:
        with pytest.raises(ValueError, match=message):
            CannyControlNetSettings(**values).validate()
