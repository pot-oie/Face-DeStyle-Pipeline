import sys
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from face_destyle.models import ModelRegistry
from face_destyle.pipelines import RegionCannyBackend, RegionCannySettings
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
  face_parsing:
    role: face_segmentation
    source: huggingface_cache
    model_id: org/parser
    revision: parser123
    license: test-only
    required_files: [config.json, preprocessor_config.json, model.safetensors]
""".strip(),
        encoding="utf-8",
    )
    hub = tmp_path / "hub"
    files = {
        "models--org--sdxl/snapshots/base123": ["model_index.json"],
        "models--org--canny/snapshots/canny123": [
            "config.json",
            "diffusion_pytorch_model.fp16.safetensors",
        ],
        "models--org--parser/snapshots/parser123": [
            "config.json",
            "preprocessor_config.json",
            "model.safetensors",
        ],
    }
    for relative, names in files.items():
        directory = hub / relative
        directory.mkdir(parents=True)
        for name in names:
            (directory / name).write_bytes(b"test")
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    return ModelRegistry.from_yaml(config), hub / "models--org--parser/snapshots/parser123"


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


def test_region_canny_saves_mask_and_attenuated_condition(tmp_path, monkeypatch):
    source = tmp_path / "input.png"
    image = Image.new("RGB", (64, 64), "black")
    draw = ImageDraw.Draw(image)
    draw.rectangle((3, 3, 18, 18), outline="white", width=3)
    draw.rectangle((24, 24, 48, 48), outline="white", width=3)
    image.save(source)
    registry, parser_path = make_registry(tmp_path, monkeypatch)
    fake_pipeline = FakePipeline()
    parser_calls = []

    def parser_factory(model_path, device):
        parser_calls.append((model_path, device))

        def parse(_image):
            labels = np.zeros((64, 64), dtype=np.uint8)
            labels[20:53, 20:53] = 1
            return labels

        return parse

    backend = RegionCannyBackend(
        RegionCannySettings(
            height=64,
            width=64,
            num_inference_steps=3,
            background_edge_scale=0.25,
            face_mask_dilation=0,
        ),
        styles_config(),
        registry,
        pipeline_factory=lambda *_args: fake_pipeline,
        face_parser_factory=parser_factory,
    )
    record = ImageRecord(
        id="sample",
        source_id="source",
        image_path=source,
        style_category="comic",
    )

    result = backend.run(record, tmp_path / "outputs", seed=42)

    assert parser_calls == [(str(parser_path), "cuda")]
    control = np.asarray(fake_pipeline.calls[0]["control_image"].convert("L"))
    assert 0 < control[:20, :20].max() < 255
    assert control[20:53, 20:53].max() == 255
    assert result.backend == "region_canny"
    assert result.extra["baseline"] == "region_canny_sdxl_controlnet_img2img"
    assert result.extra["face_parsing_revision"] == "parser123"
    assert result.extra["background_edge_scale"] == 0.25
    assert result.extra["face_mask_fraction"] == 33 * 33 / (64 * 64)
    assert result.extra["control_image_path"].endswith("sample.region-canny.png")
    assert result.extra["global_canny_image_path"].endswith("sample.canny.png")
    for suffix in ("canny", "face-mask", "region-canny"):
        assert (tmp_path / f"outputs/sample.{suffix}.png").is_file()


def test_region_canny_settings_reject_invalid_values():
    for values, message in [
        ({"background_edge_scale": 1.0}, "background_edge_scale"),
        ({"face_mask_dilation": 2}, "face_mask_dilation"),
    ]:
        try:
            RegionCannySettings(**values).validate()
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"expected invalid settings: {values}")


def test_real_parser_loader_does_not_require_auto_processor_metadata(monkeypatch):
    calls = []

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("processor", path, kwargs))
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("model", path, kwargs))
            return cls()

        def to(self, device):
            calls.append(("to", device))
            return self

        def eval(self):
            calls.append(("eval",))
            return self

    fake_torch = ModuleType("torch")
    fake_transformers = ModuleType("transformers")
    fake_transformers.SegformerImageProcessor = FakeProcessor
    fake_transformers.SegformerForSemanticSegmentation = FakeModel
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    backend = object.__new__(RegionCannyBackend)
    parser = backend._load_real_face_parser("/models/parser", "cuda")

    assert callable(parser)
    assert calls == [
        ("processor", "/models/parser", {"local_files_only": True}),
        ("model", "/models/parser", {"local_files_only": True}),
        ("to", "cuda"),
        ("eval",),
    ]
