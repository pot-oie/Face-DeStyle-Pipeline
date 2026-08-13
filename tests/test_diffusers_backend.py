from types import SimpleNamespace

import pytest
from PIL import Image

from face_destyle.pipelines.diffusers_backend import DiffusersBackend, DiffusersSettings
from face_destyle.schemas import ImageRecord


class FakePipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=[Image.new("RGB", (64, 64), (12, 34, 56))])


def styles_config():
    return {
        "styles": {
            "comic": {
                "stage1_prompt": "adaptive realistic portrait prompt",
                "negative_prompt": "comic, line art",
            }
        }
    }


def test_mock_pipeline_runs_without_downloading_or_gpu(tmp_path):
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
    backend = DiffusersBackend(settings, styles_config(), pipeline_factory=factory)
    record = ImageRecord(
        id="sample",
        source_id="source",
        image_path=source,
        style_category="comic",
    )

    result = backend.run(record, tmp_path / "outputs", seed=123)
    backend.run(record, tmp_path / "outputs-2", seed=123)

    assert len(factory_calls) == 1
    assert factory_calls[0][0] == "stabilityai/stable-diffusion-xl-base-1.0"
    assert factory_calls[0][1]["use_safetensors"] is True
    assert factory_calls[0][1]["local_files_only"] is True
    assert result.output_path.exists()
    assert result.output_path.suffix == ".png"
    assert result.backend == "diffusers"
    assert result.prompt == "adaptive realistic portrait prompt"
    assert result.extra["baseline"] == "prompt_only_sdxl_img2img"
    assert result.extra["model_id"] == settings.model_id
    assert result.extra["local_files_only"] is True
    assert fake.calls[0]["negative_prompt"] == "comic, line art"
    assert fake.calls[0]["strength"] == 0.4
    assert fake.calls[0]["image"].size == (64, 64)
    assert fake.calls[0]["generator"] == 123


def test_generic_prompt_mode_does_not_require_known_style(tmp_path):
    source = tmp_path / "input.png"
    Image.new("RGB", (64, 64)).save(source)
    fake = FakePipeline()
    backend = DiffusersBackend(
        DiffusersSettings(height=64, width=64, prompt_mode="generic"),
        {"styles": {}},
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


def test_settings_reject_non_cuda_and_batching():
    for values, message in [
        ({"device": "cpu"}, "device=cuda"),
        ({"batch_size": 2}, "batch_size must be 1"),
        ({"strength": 0.0}, "strength"),
    ]:
        try:
            DiffusersSettings(**values).validate()
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError("invalid settings were accepted")


def test_real_cache_requires_hf_home(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    with pytest.raises(RuntimeError, match="HF_HOME must point"):
        DiffusersBackend._hf_cache_dir()


def test_hub_cache_must_be_inside_hf_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "elsewhere"))
    with pytest.raises(RuntimeError, match="inside HF_HOME"):
        DiffusersBackend._hf_cache_dir()


def test_official_hf_hub_cache_variable_is_supported(tmp_path, monkeypatch):
    hf_home = tmp_path / "hf-home"
    expected = hf_home / "custom-hub"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_HUB_CACHE", str(expected))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hf_home / "legacy-hub"))

    assert DiffusersBackend._hf_cache_dir() == expected
