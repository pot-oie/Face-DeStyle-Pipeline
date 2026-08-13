"""Declarative model locations without importing GPU libraries or downloading weights."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AssetCheck:
    name: str
    location: Path | None
    missing_files: tuple[str, ...]
    available: bool
    reason: str | None = None


@dataclass(frozen=True)
class ModelAsset:
    name: str
    role: str
    source: str
    required_files: tuple[str, ...]
    license: str
    model_id: str | None = None
    revision: str | None = None
    relative_path: Path | None = None
    loader: str = "from_pretrained"
    trust_remote_code: bool = False

    @classmethod
    def from_mapping(cls, name: str, values: dict[str, Any]) -> ModelAsset:
        return cls(
            name=name,
            role=str(values["role"]),
            source=str(values["source"]),
            required_files=tuple(str(item) for item in values.get("required_files", [])),
            license=str(values.get("license", "unspecified")),
            model_id=values.get("model_id"),
            revision=values.get("revision"),
            relative_path=(
                Path(values["relative_path"]) if values.get("relative_path") else None
            ),
            loader=str(values.get("loader", "from_pretrained")),
            trust_remote_code=bool(values.get("trust_remote_code", False)),
        )


class ModelRegistry:
    """Resolve model assets against a server root or a Hugging Face cache."""

    def __init__(self, assets: dict[str, ModelAsset], *, root_env: str) -> None:
        self.assets = assets
        self.root_env = root_env

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelRegistry:
        with Path(path).open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        assets = {
            name: ModelAsset.from_mapping(name, values)
            for name, values in payload.get("assets", {}).items()
        }
        return cls(assets, root_env=str(payload.get("root_env", "FACE_DESTYLE_ROOT")))

    def require(self, name: str) -> ModelAsset:
        try:
            return self.assets[name]
        except KeyError as exc:
            raise KeyError(f"unknown model asset: {name}") from exc

    def resolve(self, name: str, *, root: str | Path | None = None) -> Path | None:
        asset = self.require(name)
        if asset.source != "local":
            return None
        selected_root = Path(root) if root is not None else self._environment_root()
        if asset.relative_path is None:
            raise ValueError(f"local asset {name} has no relative_path")
        return selected_root.expanduser().resolve() / asset.relative_path

    def check(self, name: str, *, root: str | Path | None = None) -> AssetCheck:
        asset = self.require(name)
        if asset.source == "huggingface_cache":
            return self._check_huggingface(asset)
        location = self.resolve(name, root=root)
        assert location is not None
        missing = tuple(item for item in asset.required_files if not (location / item).is_file())
        return AssetCheck(name, location, missing, not missing)

    def check_all(self, *, root: str | Path | None = None) -> list[AssetCheck]:
        return [self.check(name, root=root) for name in sorted(self.assets)]

    def _environment_root(self) -> Path:
        value = os.environ.get(self.root_env)
        if not value:
            raise RuntimeError(f"{self.root_env} must point to the persistent model root")
        return Path(value)

    @staticmethod
    def _check_huggingface(asset: ModelAsset) -> AssetCheck:
        hf_home = os.environ.get("HF_HOME")
        if not hf_home:
            return AssetCheck(asset.name, None, asset.required_files, False, "HF_HOME is not set")
        if not asset.model_id or not asset.revision:
            return AssetCheck(
                asset.name,
                None,
                asset.required_files,
                False,
                "model_id and revision are required for cached assets",
            )
        hub_value = (
            os.environ.get("HF_HUB_CACHE")
            or os.environ.get("HUGGINGFACE_HUB_CACHE")
            or Path(hf_home) / "hub"
        )
        hub = Path(hub_value)
        repository = "models--" + asset.model_id.replace("/", "--")
        snapshot = hub / repository / "snapshots" / asset.revision
        missing = tuple(item for item in asset.required_files if not (snapshot / item).is_file())
        return AssetCheck(asset.name, snapshot, missing, not missing)
