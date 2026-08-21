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
from face_destyle.filtering.prompt_rewriter import select_prompt
from face_destyle.pipelines.flux_kontext_backend import (
    FluxKontextBackend,
    FluxKontextSettings,
)
from face_destyle.schemas import DestylizationRecord, ImageRecord
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
        raise ValueError("manifest lacks required styles: " + ", ".join(missing))
    if stage in {"batch", "pilot"}:
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


def validate_resume_state(
    *,
    records_path: Path,
    failures_path: Path,
    output_dir: Path,
    selected: list[ImageRecord],
    settings: FluxKontextSettings,
    styles_config: dict,
    seed: int,
) -> set[str]:
    """Validate every persisted success and failure before resuming an interrupted run."""
    expected = {record.source_id: record for record in selected}
    completed: dict[str, DestylizationRecord] = {}
    if records_path.is_file():
        for line_number, line in enumerate(
            records_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = DestylizationRecord.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(f"invalid resume record at line {line_number}") from exc
            if record.source_id not in expected:
                raise ValueError(f"resume record is outside selected split: {record.source_id}")
            if record.source_id in completed:
                raise ValueError(f"duplicate resume success record: {record.source_id}")
            source = expected[record.source_id]
            destination = Path(record.output_path).resolve()
            required_extra = {
                "resolved_model_path": str(settings.model_dir.resolve()),
                "download_manifest": str(settings.download_manifest.resolve()),
                "hash_manifest": str(settings.hash_manifest.resolve()),
                "source_revision": settings.source_revision,
                "dtype": settings.dtype,
                "batch_size": settings.batch_size,
                "height": settings.height,
                "width": settings.width,
                "guidance_scale": settings.guidance_scale,
                "num_inference_steps": settings.num_inference_steps,
                "offload": "enable_model_cpu_offload",
                "local_files_only": settings.local_files_only,
            }
            if (
                record.id != source.id
                or Path(record.input_path).resolve() != Path(source.image_path).resolve()
                or destination.parent != output_dir.resolve()
                or destination.name != f"{source.id}.png"
                or not destination.is_file()
                or record.style_category != source.style_category
                or record.backend != FluxKontextBackend.name
                or record.seed != seed
                or record.prompt
                != select_prompt(source.style_category, styles_config, adaptive=True)
                or any(record.extra.get(key) != value for key, value in required_extra.items())
            ):
                raise ValueError(
                    f"resume success record failed frozen validation: {record.source_id}"
                )
            completed[record.source_id] = record

    declared_outputs = {Path(record.output_path).resolve() for record in completed.values()}
    if output_dir.exists():
        unexplained = [
            path
            for path in output_dir.iterdir()
            if not path.is_file() or path.resolve() not in declared_outputs
        ]
        if unexplained:
            raise ValueError(f"resume output directory contains unexplained path: {unexplained[0]}")

    if failures_path.is_file():
        for line_number, line in enumerate(
            failures_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                failure = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid resume failure at line {line_number}") from exc
            source_id = str(failure.get("source_id", ""))
            if (
                source_id not in expected
                or failure.get("backend") != FluxKontextBackend.name
                or failure.get("seed") != seed
                or not failure.get("failure_stage")
                or not failure.get("exception_type")
            ):
                raise ValueError(f"resume failure record failed validation at line {line_number}")
    return set(completed)


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
        "--split",
        choices=("pilot", "calibration", "test", "extension"),
        default="pilot",
        help="Frozen manifest split to load; test must remain sealed until calibration is frozen.",
    )
    parser.add_argument(
        "--probe-stage",
        choices=("first", "remaining", "all", "batch", "pilot"),
        default="first",
        help="batch runs every record in the selected split; pilot is its legacy alias.",
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
        split=args.split,
    )
    selected = select_probe_records(manifest_records, args.probe_stage)
    settings = FluxKontextSettings(
        model_dir=args.model_dir.resolve(),
        download_manifest=args.download_manifest.resolve(),
        hash_manifest=args.hash_manifest.resolve(),
        source_revision=args.source_revision,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )
    styles_config = load_yaml(args.styles_config)
    if args.resume:
        try:
            completed = validate_resume_state(
                records_path=args.records_output,
                failures_path=args.failures_output,
                output_dir=args.output_dir,
                selected=selected,
                settings=settings,
                styles_config=styles_config,
                seed=args.seed,
            )
        except ValueError as exc:
            parser.error(str(exc))
        selected = [record for record in selected if record.source_id not in completed]
        print(
            f"Resume validation passed; skipping {len(completed)} completed source IDs",
            flush=True,
        )
    if not selected:
        print("No pending records; probe stage is already complete")
        return 0
    seed_everything(args.seed)
    backend = FluxKontextBackend(settings, styles_config)
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
