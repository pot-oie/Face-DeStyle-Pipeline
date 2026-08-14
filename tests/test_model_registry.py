from pathlib import Path

import pytest

from face_destyle.models import ModelRegistry


def write_config(path: Path) -> None:
    path.write_text(
        """
root_env: TEST_MODEL_ROOT
assets:
  local_metric:
    role: metric
    source: local
    relative_path: models/metric
    license: Apache-2.0
    required_files: [config.json, model.safetensors]
  cached_model:
    role: generator
    source: huggingface_cache
    model_id: org/model
    revision: abc123
    license: test-only
    required_files: [config.json]
""".strip(),
        encoding="utf-8",
    )


def test_local_asset_check_is_offline(tmp_path):
    config = tmp_path / "models.yaml"
    write_config(config)
    model = tmp_path / "models/metric"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"not-loaded-by-registry")

    check = ModelRegistry.from_yaml(config).check("local_metric", root=tmp_path)

    assert check.available
    assert check.location == model
    assert check.missing_files == ()


def test_local_asset_reports_missing_files(tmp_path):
    config = tmp_path / "models.yaml"
    write_config(config)
    (tmp_path / "models/metric").mkdir(parents=True)

    check = ModelRegistry.from_yaml(config).check("local_metric", root=tmp_path)

    assert not check.available
    assert check.missing_files == ("config.json", "model.safetensors")


def test_cached_asset_resolves_pinned_snapshot(tmp_path, monkeypatch):
    config = tmp_path / "models.yaml"
    write_config(config)
    snapshot = tmp_path / "hf/hub/models--org--model/snapshots/abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)

    check = ModelRegistry.from_yaml(config).check("cached_model")

    assert check.available
    assert check.location == snapshot
    assert ModelRegistry.from_yaml(config).resolve("cached_model") == snapshot


def test_registry_supports_official_hf_hub_cache_variable(tmp_path, monkeypatch):
    config = tmp_path / "models.yaml"
    write_config(config)
    custom_hub = tmp_path / "custom-hub"
    snapshot = custom_hub / "models--org--model/snapshots/abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HF_HUB_CACHE", str(custom_hub))

    assert ModelRegistry.from_yaml(config).check("cached_model").available


def test_hf_hub_cache_is_sufficient_without_hf_home(tmp_path, monkeypatch):
    config = tmp_path / "models.yaml"
    write_config(config)
    custom_hub = tmp_path / "custom-hub"
    snapshot = custom_hub / "models--org--model/snapshots/abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(custom_hub))

    assert ModelRegistry.from_yaml(config).resolve("cached_model") == snapshot


def test_missing_root_environment_is_friendly(tmp_path, monkeypatch):
    config = tmp_path / "models.yaml"
    write_config(config)
    monkeypatch.delenv("TEST_MODEL_ROOT", raising=False)

    check = ModelRegistry.from_yaml(config).check("local_metric")

    assert not check.available
    assert check.reason is not None
    assert "TEST_MODEL_ROOT" in check.reason


def test_local_asset_cannot_escape_model_root(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text(
        """
root_env: TEST_MODEL_ROOT
assets:
  escaped:
    role: metric
    source: local
    relative_path: ../outside
    license: test-only
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes"):
        ModelRegistry.from_yaml(config).resolve("escaped", root=tmp_path / "root")


def test_repository_buffalo_l_registry_requires_all_five_onnx_files():
    asset = ModelRegistry.from_yaml("configs/models.yaml").require("insightface_buffalo_l")

    assert set(asset.required_files) == {
        "det_10g.onnx",
        "2d106det.onnx",
        "1k3d68.onnx",
        "genderage.onnx",
        "w600k_r50.onnx",
    }
