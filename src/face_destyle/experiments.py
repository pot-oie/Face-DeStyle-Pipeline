"""Validated declarations for planned ablations; declarations are not experimental results."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    settings: dict[str, Any]
    extension: bool = False


@dataclass(frozen=True)
class ExperimentRun:
    run_id: str
    experiment: str
    seed: int
    extension: bool
    settings: dict[str, Any]


def load_experiment_specs(path: str | Path) -> list[ExperimentSpec]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    defaults = dict(payload.get("defaults", {}))
    specs = []
    for section, extension in (("experiments", False), ("extensions", True)):
        for name, values in payload.get(section, {}).items():
            specs.append(
                ExperimentSpec(name=name, settings=defaults | dict(values), extension=extension)
            )
    return specs


def load_evaluation_assets(path: str | Path) -> set[str]:
    """Return every model asset named by primary or robustness evaluation configuration."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    evaluation = payload.get("evaluation", {})
    return {
        str(asset)
        for group in evaluation.values()
        for asset in group.values()
        if asset is not None
    }


def referenced_assets(spec: ExperimentSpec) -> set[str]:
    """Return model-registry names referenced by a declared experiment."""
    model_keys = {
        "generator",
        "control_model",
        "pose_extractor",
        "depth_extractor",
        "refiner",
        "vae",
        "identity_control",
    }
    names = {str(value) for key, value in spec.settings.items() if key in model_keys}
    names.update(str(value) for value in spec.settings.get("control_models", []))
    face_region_mode = spec.settings.get("face_region_mode")
    if face_region_mode not in {None, "none"}:
        names.add(str(face_region_mode))
    return names


def validate_asset_references(specs: list[ExperimentSpec], asset_names: set[str]) -> None:
    errors = []
    for spec in specs:
        unknown = referenced_assets(spec) - asset_names
        if unknown:
            errors.append(f"{spec.name}: {', '.join(sorted(unknown))}")
    if errors:
        raise ValueError("unknown model assets in experiments: " + "; ".join(errors))


def expand_runs(specs: list[ExperimentSpec], seeds: list[int]) -> list[ExperimentRun]:
    """Expand declarations into deterministic, auditable run identifiers."""
    runs = []
    for spec in specs:
        for seed in seeds:
            settings = dict(spec.settings) | {"seed": seed}
            canonical = json.dumps(
                {"experiment": spec.name, "settings": settings},
                sort_keys=True,
                separators=(",", ":"),
            )
            run_id = f"{spec.name}-{hashlib.sha256(canonical.encode()).hexdigest()[:12]}"
            runs.append(ExperimentRun(run_id, spec.name, seed, spec.extension, settings))
    return runs
