from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from face_destyle.models import ModelRegistry
from face_destyle.pipelines.diffusers_backend import DiffusersBackend, DiffusersSettings
from face_destyle.schemas import ImageRecord
from face_destyle.utils.io import load_yaml

ROOT = Path(__file__).resolve().parents[1]


class FakePipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=[Image.new("RGB", (64, 64), (12, 34, 56))])


def make_registry(tmp_path, monkeypatch):
    config = tmp_path / "models.yaml"
    config.write_text(
        """
assets:
  sdxl_base:
    role: generator
    source: huggingface_cache
    model_id: org/sdxl
    revision: pinned123
    license: test-only
    required_files: [model_index.json]
""".strip(),
        encoding="utf-8",
    )
    hub = tmp_path / "hub"
    snapshot = hub / "models--org--sdxl/snapshots/pinned123"
    snapshot.mkdir(parents=True)
    (snapshot / "model_index.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    return ModelRegistry.from_yaml(config), snapshot


def styles_config():
    return {
        "styles": {
            "comic": {
                "stage1_prompt": "adaptive realistic portrait prompt",
                "negative_prompt": "comic, line art",
            }
        }
    }


def test_repository_config_uses_registry_as_model_source_of_truth():
    config = load_yaml(ROOT / "configs/inference.yaml")

    assert config["model_asset"] == "sdxl_base"
    assert "model_id" not in config
    assert "revision" not in config


def test_mock_pipeline_runs_without_downloading_or_gpu(tmp_path, monkeypatch):
    source = tmp_path / "input.jpg"
    Image.new("RGB", (80, 100), (100, 110, 120)).save(source)
    fake = FakePipeline()
    factory_calls = []

    def factory(model_id, kwargs):
        factory_calls.append((model_id, kwargs))
        return fake

    settings = DiffusersSettings(
        height=64,
        width=64,
        num_inference_steps=3,
        strength=0.4,
    )
    registry, snapshot = make_registry(tmp_path, monkeypatch)
    backend = DiffusersBackend(settings, styles_config(), registry, pipeline_factory=factory)
    record = ImageRecord(
        id="sample",
        source_id="source",
        image_path=source,
        style_category="comic",
    )

    result = backend.run(record, tmp_path / "outputs", seed=123)
    backend.run(record, tmp_path / "outputs-2", seed=123)

    assert len(factory_calls) == 1
    assert factory_calls[0][0] == str(snapshot)
    assert factory_calls[0][1]["use_safetensors"] is True
    assert factory_calls[0][1]["local_files_only"] is True
    assert "revision" not in factory_calls[0][1]
    assert result.output_path.exists()
    assert result.output_path.suffix == ".png"
    assert result.backend == "diffusers"
    assert result.prompt == "adaptive realistic portrait prompt"
    assert result.extra["baseline"] == "prompt_only_sdxl_img2img"
    assert result.extra["model_asset"] == "sdxl_base"
    assert result.extra["model_id"] == "org/sdxl"
    assert result.extra["revision"] == "pinned123"
    assert result.extra["resolved_model_path"] == str(snapshot)
    assert result.extra["local_files_only"] is True
    assert result.extra["pipeline_loaded_this_run"] is True
    assert result.extra["inference_seconds"] >= 0
    assert fake.calls[0]["negative_prompt"] == "comic, line art"
    assert fake.calls[0]["strength"] == 0.4
    assert fake.calls[0]["image"].size == (64, 64)
    assert fake.calls[0]["generator"] == 123


def test_generic_prompt_mode_does_not_require_known_style(tmp_path, monkeypatch):
    source = tmp_path / "input.png"
    Image.new("RGB", (64, 64)).save(source)
    fake = FakePipeline()
    registry, _snapshot = make_registry(tmp_path, monkeypatch)
    backend = DiffusersBackend(
        DiffusersSettings(height=64, width=64, prompt_mode="generic"),
        {"styles": {}},
        registry,
        pipeline_factory=lambda _model_id, _kwargs: fake,
    )
    record = ImageRecord(
        id="sample",
        source_id="sample",
        image_path=source,
        style_category="unknown",
    )

    result = backend.run(record, tmp_path / "output", seed=7)

    assert "natural realistic photograph" in result.prompt
    assert fake.calls[0]["negative_prompt"] == ""


def test_settings_reject_invalid_runtime_values():
    for values, message in [
        ({"device": "cpu"}, "device=cuda"),
        ({"batch_size": 2}, "batch_size must be 1"),
        ({"strength": 0.0}, "strength"),
        ({"local_files_only": False}, "local_files_only=true"),
    ]:
        try:
            DiffusersSettings(**values).validate()
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError("invalid settings were accepted")


def test_unavailable_registered_model_is_rejected(tmp_path, monkeypatch):
    registry, snapshot = make_registry(tmp_path, monkeypatch)
    (snapshot / "model_index.json").unlink()
    backend = DiffusersBackend(
        DiffusersSettings(height=64, width=64),
        styles_config(),
        registry,
        pipeline_factory=lambda _path, _kwargs: FakePipeline(),
    )
    source = tmp_path / "input.png"
    Image.new("RGB", (64, 64)).save(source)
    record = ImageRecord(id="sample", source_id="sample", image_path=source, style_category="comic")

    with pytest.raises(RuntimeError, match="model_index.json"):
        backend.run(record, tmp_path / "output", seed=7)


def test_existing_output_is_not_overwritten(tmp_path, monkeypatch):
    registry, _snapshot = make_registry(tmp_path, monkeypatch)
    backend = DiffusersBackend(
        DiffusersSettings(height=64, width=64),
        styles_config(),
        registry,
        pipeline_factory=lambda _path, _kwargs: FakePipeline(),
    )
    source = tmp_path / "input.png"
    Image.new("RGB", (64, 64)).save(source)
    destination = tmp_path / "output/sample.png"
    destination.parent.mkdir()
    destination.write_bytes(b"keep")
    record = ImageRecord(id="sample", source_id="sample", image_path=source, style_category="comic")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        backend.run(record, destination.parent, seed=7)

    assert destination.read_bytes() == b"keep"
