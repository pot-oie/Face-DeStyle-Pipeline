#!/usr/bin/env python3
"""Build a small synthetic paired dataset for a 3D-destylization LoRA smoke test."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

PORTRAIT_SPECS = (
    "young Black woman with short natural curls, calm expression, warm home interior",
    "older East Asian man with swept gray hair, thoughtful expression, ceramic studio",
    "middle-aged white woman with tied red hair, curious upward gaze, greenhouse",
    "young Black man with a short fade haircut, alert side gaze, evening train interior",
    "young Middle Eastern man with a shaved head, slight smile, sunlit terrace",
    "South Asian woman in her thirties with long wavy hair, direct gaze, book-lined room",
    "Latino man in his forties with salt-and-pepper beard, relaxed expression, cafe",
    "East Asian woman in her twenties with a bob haircut, gentle smile, bright kitchen",
    "older Black woman with silver braids, dignified expression, neutral studio backdrop",
    "white man in his twenties with curly blond hair, surprised expression, workshop",
    "South Asian man in his sixties with gray beard, three-quarter view, garden",
    "Latina woman in her fifties with short dark hair, broad smile, sunlit living room",
    "young Southeast Asian man with textured crop haircut, neutral expression, library",
    "Middle Eastern woman in her thirties with loosely wrapped scarf, side gaze, courtyard",
    "older white woman with cropped silver hair, serious expression, painter's studio",
    "young Black woman with long locs, subtle smile, modern office",
    "East Asian man in his forties with round glasses, calm expression, study room",
    "South Asian woman in her sixties with gray bun, warm expression, veranda",
    "young Latino man with wavy hair and light stubble, direct gaze, city apartment",
    "white woman in her thirties with freckles and auburn hair, laughing, flower shop",
    "older Middle Eastern man with white beard, profile view, softly lit workshop",
    "young Southeast Asian woman with long straight hair, focused expression, music room",
    "Black man in his fifties with close-cropped gray hair, gentle smile, gallery interior",
    "androgynous young adult with short dark curls, neutral expression, daylight studio",
)

DESTYLIZE_INSTRUCTION = (
    "Convert this 3D cartoon portrait into a natural realistic photograph. Remove exaggerated "
    "facial geometry, oversized eyes, plastic skin, and CGI rendering while preserving the "
    "person, pose, expression, clothing, composition, and background."
)

STYLIZE_INSTRUCTION = (
    "Transform this portrait into a polished 3D animated-film character with clearly exaggerated "
    "facial geometry, slightly oversized expressive eyes, smooth plastic-like skin, and cinematic "
    "CGI lighting. Preserve the person, pose, expression, clothing, composition, and background."
)


def selected_indices(start_index: int, count: int) -> range:
    if start_index < 1 or count < 1:
        raise ValueError("start-index and count must be positive")
    stop = start_index + count
    if stop > len(PORTRAIT_SPECS) + 1:
        raise ValueError(f"requested portraits exceed the available {len(PORTRAIT_SPECS)} specs")
    return range(start_index, stop)


def portrait_name(index: int) -> str:
    return f"portrait-{index:03d}.png"


def natural_prompt(index: int) -> str:
    spec = PORTRAIT_SPECS[index - 1]
    return (
        "Photorealistic head-and-shoulders portrait of a fictional person: "
        f"{spec}. Natural facial anatomy and skin texture, realistic hair, documentary "
        "photography, 50mm lens, coherent background, no illustration and no CGI."
    )


def generate_targets(
    *, model_path: Path, output_dir: Path, indices: range, seed: int
) -> None:
    import torch
    from diffusers import StableDiffusionXLPipeline

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        variant="fp16",
        local_files_only=True,
    )
    pipeline.enable_model_cpu_offload()
    target_dir = output_dir / "train" / "target"
    target_dir.mkdir(parents=True, exist_ok=True)
    negative = (
        "illustration, cartoon, anime, 3d render, cgi, doll, plastic skin, deformed face, "
        "duplicate person, text, watermark"
    )
    for index in indices:
        destination = target_dir / portrait_name(index)
        if destination.exists():
            print(f"SKIP target {destination.name}", flush=True)
            continue
        generator = torch.Generator(device="cpu").manual_seed(seed + index)
        image = pipeline(
            prompt=natural_prompt(index),
            negative_prompt=negative,
            height=1024,
            width=1024,
            num_inference_steps=28,
            guidance_scale=5.0,
            generator=generator,
        ).images[0]
        image.convert("RGB").save(destination)
        print(f"OK target {destination.name}", flush=True)
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


def generate_conditions(
    *, model_path: Path, output_dir: Path, indices: range, seed: int
) -> None:
    import torch
    from diffusers import FluxKontextPipeline
    from PIL import Image

    pipeline = FluxKontextPipeline.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16, local_files_only=True
    )
    pipeline.enable_model_cpu_offload()
    target_dir = output_dir / "train" / "target"
    condition_dir = output_dir / "train" / "condition"
    condition_dir.mkdir(parents=True, exist_ok=True)
    for index in indices:
        name = portrait_name(index)
        source = target_dir / name
        destination = condition_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"target must be generated first: {source}")
        if destination.exists():
            print(f"SKIP condition {destination.name}", flush=True)
            continue
        with Image.open(source) as image:
            initial = image.convert("RGB")
        generator = torch.Generator(device="cpu").manual_seed(seed + index)
        result = pipeline(
            image=initial,
            prompt=STYLIZE_INSTRUCTION,
            guidance_scale=2.5,
            num_inference_steps=28,
            generator=generator,
            height=1024,
            width=1024,
            max_area=1024 * 1024,
        ).images[0]
        result.convert("RGB").save(destination)
        print(f"OK condition {destination.name}", flush=True)
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


def write_metadata(*, output_dir: Path, indices: range) -> Path:
    train_dir = output_dir / "train"
    rows = []
    for index in indices:
        name = portrait_name(index)
        target = train_dir / "target" / name
        condition = train_dir / "condition" / name
        if not target.is_file() or not condition.is_file():
            raise FileNotFoundError(f"pair is incomplete: {name}")
        rows.append(
            {
                "target_file_name": f"target/{name}",
                "condition_file_name": f"condition/{name}",
                "instruction": DESTYLIZE_INSTRUCTION,
            }
        )
    destination = train_dir / "metadata.jsonl"
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"WROTE {len(rows)} pairs -> {destination}", flush=True)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sdxl-model", type=Path)
    parser.add_argument("--flux-model", type=Path)
    parser.add_argument(
        "--stage", choices=("targets", "conditions", "metadata", "all"), default="all"
    )
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--target-seed", type=int, default=20260823)
    parser.add_argument("--condition-seed", type=int, default=20260824)
    args = parser.parse_args()

    try:
        indices = selected_indices(args.start_index, args.count)
    except ValueError as exc:
        parser.error(str(exc))
    if args.stage in {"targets", "all"}:
        if args.sdxl_model is None:
            parser.error("--sdxl-model is required for target generation")
        generate_targets(
            model_path=args.sdxl_model,
            output_dir=args.output_dir,
            indices=indices,
            seed=args.target_seed,
        )
    if args.stage in {"conditions", "all"}:
        if args.flux_model is None:
            parser.error("--flux-model is required for condition generation")
        generate_conditions(
            model_path=args.flux_model,
            output_dir=args.output_dir,
            indices=indices,
            seed=args.condition_seed,
        )
    if args.stage in {"metadata", "all"}:
        write_metadata(output_dir=args.output_dir, indices=indices)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
