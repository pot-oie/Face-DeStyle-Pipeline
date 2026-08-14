#!/usr/bin/env python3
"""Run a matched multi-image img2img strength sweep while reusing one loaded pipeline."""

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

from tqdm import tqdm

from face_destyle.data.manifests import load_dataset_manifest
from face_destyle.data.metadata import write_jsonl
from face_destyle.models import ModelRegistry
from face_destyle.pipelines import (
    CannyControlNetBackend,
    CannyControlNetSettings,
    DiffusersBackend,
    DiffusersSettings,
)
from face_destyle.utils.io import load_yaml
from face_destyle.utils.reproducibility import seed_everything


def strength_tag(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--strength", type=float, nargs="+", required=True)
    parser.add_argument("--backend", choices=("diffusers", "canny"), default="diffusers")
    parser.add_argument("--prompt-mode", choices=("generic", "adaptive"), default="adaptive")
    parser.add_argument("--split", choices=("pilot", "calibration", "test"), default="pilot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--styles-config", type=Path, default=Path("configs/styles.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    args = parser.parse_args()

    if len(set(args.strength)) != len(args.strength):
        parser.error("--strength values must be unique")
    if any(not 0.0 < value <= 1.0 for value in args.strength):
        parser.error("--strength values must be in (0, 1]")
    if args.output_root.exists():
        if not args.output_root.is_dir():
            parser.error(f"output root is not a directory: {args.output_root}")
        if any(args.output_root.iterdir()):
            parser.error(f"refusing to use non-empty output root: {args.output_root}")

    records = load_dataset_manifest(
        args.manifest,
        data_root=args.data_root,
        split=args.split,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.manifest, args.output_root / "runtime_manifest.jsonl")
    (args.output_root / "sweep.json").write_text(
        json.dumps(
            {
                "backend": args.backend,
                "prompt_mode": args.prompt_mode,
                "strengths": args.strength,
                "seed": args.seed,
                "split": args.split,
                "record_count": len(records),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    config = load_yaml(args.config)
    config["prompt_mode"] = args.prompt_mode
    config["strength"] = args.strength[0]
    styles_config = load_yaml(args.styles_config)
    model_registry = ModelRegistry.from_yaml(args.models_config)
    if args.backend == "canny":
        settings = CannyControlNetSettings.from_mapping(config)
        backend = CannyControlNetBackend(settings, styles_config, model_registry)
    else:
        settings = DiffusersSettings.from_mapping(config)
        backend = DiffusersBackend(settings, styles_config, model_registry)

    seed_everything(args.seed)
    for strength in args.strength:
        backend.settings = replace(backend.settings, strength=strength)
        backend.settings.validate()
        run_name = f"{args.backend}-{args.prompt_mode}-strength-{strength_tag(strength)}"
        run_dir = args.output_root / run_name
        image_dir = run_dir / "images"
        records_path = run_dir / "records.jsonl"
        outputs = []
        for record in tqdm(records, desc=run_name):
            outputs.append(backend.run(record, image_dir, seed=args.seed))
            write_jsonl(outputs, records_path)
        print(f"Completed {run_name}: {len(outputs)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
