#!/usr/bin/env python3
"""List declared primary and extension experiments without running inference."""

import argparse
import dataclasses
import json
from pathlib import Path

from face_destyle.experiments import (
    expand_runs,
    load_evaluation_assets,
    load_experiment_specs,
    validate_asset_references,
)
from face_destyle.models import ModelRegistry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--json", action="store_true", help="Emit JSON lines.")
    parser.add_argument("--seed", type=int, action="append", help="Expand runs for this seed.")
    args = parser.parse_args()
    specs = load_experiment_specs(args.config)
    registry = ModelRegistry.from_yaml(args.models_config)
    validate_asset_references(specs, set(registry.assets))
    unknown_evaluators = load_evaluation_assets(args.config) - set(registry.assets)
    if unknown_evaluators:
        parser.error(f"unknown evaluation assets: {', '.join(sorted(unknown_evaluators))}")
    if args.seed:
        for run in expand_runs(specs, args.seed):
            print(json.dumps(dataclasses.asdict(run), sort_keys=True))
        return 0
    for spec in specs:
        payload = {"name": spec.name, "extension": spec.extension, **spec.settings}
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            kind = "extension" if spec.extension else "primary"
            print(f"{kind:9} {spec.name:28} {spec.settings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
