#!/usr/bin/env python3
"""Run the native-1024 FLUX.1 Kontext capability probe."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import defaultdict
from pathlib import Path

from face_destyle.data.manifests import load_dataset_manifest
from face_destyle.pipelines.flux_kontext_backend import (
    FluxKontextBackend,
    FluxKontextSettings,
)
from face_destyle.schemas import ImageRecord
from face_destyle.utils.io import load_yaml
from face_destyle.utils.reproducibility import seed_everything

REQUIRED_STYLES = ("3d_cartoon", "comic", "ink", "watercolor")


def select_probe_records(
    records: list[ImageRecord], stage: str
) -> list[ImageRecord]:
    grouped: defaultdict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.style_category].append(record)
    missing = [style for style in REQUIRED_STYLES if not grouped[style]]
    if missing:
        raise ValueError("manifest lacks accepted pilot styles: " + ", ".join(missing))
    if stage == "pilot":
        style_order = {style: index for index, style in enumerate(REQUIRED_STYLES)}
        return sorted(
            records,
            key=lambda item: (style_order[item.style_category], item.source_id),
        )
    selected = [
        sorted(grouped[style], key=lambda item: item.source_id)[0]
        for style in REQUIRED_STYLES
    ]
    if stage == "first":
        return selected[:1]
    if stage == "remaining":
        return selected[1:]
    return selected


def completed_source_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            completed.add(str(json.loads(line)["source_id"]))
    return completed


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        if hasattr(payload, "model_dump_json"):
            handle.write(payload.model_dump_json())
        else:
            handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\n")
        handle.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--hash-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--records-output", type=Path, required=True)
    parser.add_argument("--failures-output", type=Path, required=True)
    parser.add_argument("--styles-config", type=Path, default=Path("configs/styles.yaml"))
    parser.add_argument("--source-revision", default="master")
    parser.add_argument(
        "--probe-stage",
        choices=("first", "remaining", "all", "pilot"),
        default="first",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an interrupted run and skip source IDs already in records-output.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    args = parser.parse_args()

    if not args.resume and (args.records_output.exists() or args.failures_output.exists()):
        parser.error("refusing to append to an existing records or failures file")
    manifest_records = load_dataset_manifest(
        args.manifest,
        data_root=args.data_root,
        split="pilot",
    )
    selected = select_probe_records(manifest_records, args.probe_stage)
    if args.resume:
        completed = completed_source_ids(args.records_output)
        selected = [record for record in selected if record.source_id not in completed]
        print(f"Resume mode: skipping {len(completed)} completed source IDs", flush=True)
    if not selected:
        print("No pending records; probe stage is already complete")
        return 0
    settings = FluxKontextSettings(
        model_dir=args.model_dir.resolve(),
        download_manifest=args.download_manifest.resolve(),
        hash_manifest=args.hash_manifest.resolve(),
        source_revision=args.source_revision,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )
    seed_everything(args.seed)
    backend = FluxKontextBackend(settings, load_yaml(args.styles_config))
    failures = 0
    for record in selected:
        started = time.perf_counter()
        try:
            result = backend.run(record, args.output_dir, seed=args.seed)
            append_jsonl(args.records_output, result)
            print(f"OK {record.source_id} -> {result.output_path}", flush=True)
        except Exception as exc:  # noqa: BLE001 - failures are experimental records
            failures += 1
            append_jsonl(
                args.failures_output,
                {
                    "id": record.id,
                    "source_id": record.source_id,
                    "style_category": record.style_category,
                    "backend": backend.name,
                    "seed": args.seed,
                    "failure_stage": (
                        "pipeline_inference" if backend.pipeline_loaded else "model_load"
                    ),
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "elapsed_seconds": time.perf_counter() - started,
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"FAILED {record.source_id}: {type(exc).__name__}: {exc}", flush=True)
            if args.probe_stage == "first":
                break
    print(
        f"Probe stage={args.probe_stage}: "
        f"success={len(selected) - failures}, failures={failures}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
