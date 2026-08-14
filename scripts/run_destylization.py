#!/usr/bin/env python3
"""Run a configured destylization backend over JSONL metadata."""

import argparse
import hashlib
from pathlib import Path
from typing import Any

from tqdm import tqdm

from face_destyle.data.manifests import load_dataset_manifest
from face_destyle.data.metadata import read_jsonl, write_jsonl
from face_destyle.models import ModelRegistry
from face_destyle.pipelines import (
    CannyControlNetBackend,
    CannyControlNetSettings,
    CopyBackend,
    DiffusersBackend,
    DiffusersSettings,
)
from face_destyle.schemas import ImageRecord
from face_destyle.utils.io import load_yaml
from face_destyle.utils.reproducibility import seed_everything


def make_backend(
    backend_name: str,
    inference_config: dict[str, Any],
    styles_config: dict[str, Any],
    model_registry: ModelRegistry,
):
    if backend_name == "copy":
        return CopyBackend()
    if backend_name == "canny":
        settings = CannyControlNetSettings.from_mapping(inference_config)
        return CannyControlNetBackend(settings, styles_config, model_registry)
    settings = DiffusersSettings.from_mapping(inference_config)
    return DiffusersBackend(settings, styles_config, model_registry)


def single_record(path: Path, style_category: str, record_id: str | None) -> ImageRecord:
    identifier = record_id or hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    return ImageRecord(
        id=identifier,
        source_id=identifier,
        image_path=path,
        style_category=style_category,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--metadata", type=Path, help="Batch input JSONL.")
    inputs.add_argument("--manifest", type=Path, help="Portable frozen dataset manifest JSONL.")
    inputs.add_argument("--input", type=Path, help="Single input image.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--records-output", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--styles-config", type=Path, default=Path("configs/styles.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--data-root", type=Path, help="Root for relative paths in --manifest.")
    parser.add_argument("--split", choices=("pilot", "calibration", "test", "extension"))
    parser.add_argument("--backend", choices=("copy", "diffusers", "canny"))
    parser.add_argument(
        "--prompt-mode",
        choices=("generic", "adaptive"),
        help="Override the configured prompt mode for a Diffusers run.",
    )
    parser.add_argument(
        "--control-scale",
        type=float,
        help="Override ControlNet conditioning scale for --backend canny.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--style-category", help="Required with --input.")
    parser.add_argument("--record-id", help="Optional stable ID override for --input.")
    args = parser.parse_args()
    if args.records_output is not None and args.records_output.exists():
        parser.error(f"refusing to overwrite existing records output: {args.records_output}")
    config = load_yaml(args.config)
    styles_config = load_yaml(args.styles_config)
    model_registry = ModelRegistry.from_yaml(args.models_config)
    backend_name = args.backend or str(config.get("backend", "copy"))
    if args.prompt_mode is not None:
        if backend_name not in {"diffusers", "canny"}:
            parser.error("--prompt-mode is only valid with --backend diffusers or canny")
        config["prompt_mode"] = args.prompt_mode
    if args.control_scale is not None:
        if backend_name != "canny":
            parser.error("--control-scale is only valid with --backend canny")
        config["controlnet_conditioning_scale"] = args.control_scale
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    seed_everything(seed)
    backend = make_backend(backend_name, config, styles_config, model_registry)
    if args.metadata or args.manifest:
        if args.records_output is None:
            parser.error("--records-output is required with batch inputs")
        if args.manifest:
            records = load_dataset_manifest(
                args.manifest,
                data_root=args.data_root,
                split=args.split,
            )
        else:
            if args.data_root is not None or args.split is not None:
                parser.error("--data-root and --split are only valid with --manifest")
            records = read_jsonl(
                args.metadata, ImageRecord, check_paths=True, path_fields=("image_path",)
            )
    else:
        if args.data_root is not None or args.split is not None:
            parser.error("--data-root and --split are only valid with --manifest")
        if not args.style_category:
            parser.error("--style-category is required with --input")
        records = [single_record(args.input, args.style_category, args.record_id)]
        if not args.input.exists():
            parser.error(f"input image does not exist: {args.input}")
    outputs = [backend.run(item, args.output_dir, seed=seed) for item in tqdm(records)]
    if args.records_output:
        write_jsonl(outputs, args.records_output)
        destination = str(args.records_output)
    else:
        destination = "image output only"
    print(f"Wrote {len(outputs)} {backend_name} records ({destination})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
