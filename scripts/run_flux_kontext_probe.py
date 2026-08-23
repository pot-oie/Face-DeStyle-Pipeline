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
from face_destyle.data.pair_bank import load_pair_bank_source_list
from face_destyle.filtering.prompt_rewriter import select_stage_prompt
from face_destyle.pipelines.flux_kontext_backend import (
    FluxKontextBackend,
    FluxKontextSettings,
)
from face_destyle.schemas import DestylizationRecord, ImageRecord
from face_destyle.utils.io import load_yaml
from face_destyle.utils.reproducibility import seed_everything

REQUIRED_STYLES = ("3d_cartoon", "comic", "ink", "watercolor")


def select_probe_records(
    records: list[ImageRecord],
    stage: str,
    *,
    required_styles: tuple[str, ...] = REQUIRED_STYLES,
) -> list[ImageRecord]:
    grouped: defaultdict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.style_category].append(record)
    if not required_styles or len(set(required_styles)) != len(required_styles):
        raise ValueError("required styles must be non-empty and unique")
    missing = [style for style in required_styles if not grouped[style]]
    if missing:
        raise ValueError("manifest lacks required styles: " + ", ".join(missing))
    if stage in {"batch", "pilot"}:
        style_order = {style: index for index, style in enumerate(required_styles)}
        return sorted(
            (record for record in records if record.style_category in style_order),
            key=lambda item: (style_order[item.style_category], item.source_id),
        )
    selected = [
        sorted(grouped[style], key=lambda item: item.source_id)[0]
        for style in required_styles
    ]
    if stage == "first":
        return selected[:1]
    if stage == "remaining":
        return selected[1:]
    return selected


def filter_records_by_source_ids(
    records: list[ImageRecord], source_ids: list[str]
) -> list[ImageRecord]:
    """Keep an explicit source subset while retaining the input record order."""
    if not source_ids:
        return records
    requested = set(source_ids)
    if len(requested) != len(source_ids):
        raise ValueError("source IDs must be unique")
    available = {record.source_id for record in records}
    missing = sorted(requested - available)
    if missing:
        raise ValueError("requested source IDs are unavailable: " + ", ".join(missing))
    return [record for record in records if record.source_id in requested]


def validate_resume_state(
    *,
    records_path: Path,
    failures_path: Path,
    output_dir: Path,
    selected: list[ImageRecord],
    settings: FluxKontextSettings,
    styles_config: dict,
    seed: int,
    prompt_stage: str = "stage1",
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
                "lora_weights": (
                    str(settings.lora_weights.resolve())
                    if settings.lora_weights is not None
                    else None
                ),
                "lora_scale": settings.lora_scale,
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
                != select_stage_prompt(
                    source.style_category,
                    styles_config,
                    stage=prompt_stage,
                )
                or record.extra.get("prompt_stage", "stage1") != prompt_stage
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
                or failure.get("prompt_stage", "stage1") != prompt_stage
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


def load_sequential_inputs(records_path: Path) -> list[ImageRecord]:
    """Use successful Stage 1 outputs as explicit inputs to a sequential edit."""
    if not records_path.is_file():
        raise FileNotFoundError(f"Stage 1 records file does not exist: {records_path}")
    records: list[ImageRecord] = []
    source_ids: set[str] = set()
    for line_number, line in enumerate(
        records_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            parent = DestylizationRecord.model_validate_json(line)
        except ValueError as exc:
            raise ValueError(f"invalid Stage 1 record at line {line_number}") from exc
        if parent.source_id in source_ids:
            raise ValueError(f"duplicate Stage 1 source_id: {parent.source_id}")
        if parent.extra.get("prompt_stage", "stage1") != "stage1":
            raise ValueError(
                f"sequential input is not a Stage 1 record: {parent.source_id}"
            )
        output_path = Path(parent.output_path).expanduser().resolve()
        if not output_path.is_file():
            raise FileNotFoundError(
                f"missing Stage 1 output for {parent.source_id}: {output_path}"
            )
        source_ids.add(parent.source_id)
        records.append(
            ImageRecord(
                id=parent.id,
                source_id=parent.source_id,
                image_path=output_path,
                style_category=parent.style_category,
            )
        )
    if not records:
        raise ValueError("Stage 1 records file contains no successful outputs")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", type=Path)
    inputs.add_argument(
        "--source-list",
        type=Path,
        help="Lightweight pair-bank CSV; only rows with role=candidate are generated.",
    )
    inputs.add_argument(
        "--input-records",
        type=Path,
        help=(
            "Successful Stage 1 DestylizationRecord JSONL. Its output images become the "
            "explicit inputs for a true sequential Stage 2 run."
        ),
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--hash-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--records-output", type=Path, required=True)
    parser.add_argument("--failures-output", type=Path, required=True)
    parser.add_argument("--styles-config", type=Path, default=Path("configs/styles.yaml"))
    parser.add_argument(
        "--required-style",
        action="append",
        dest="required_styles",
        help=(
            "Style required and selected from the manifest; repeat for multiple styles. "
            "Defaults to the four legacy formal-v1 styles."
        ),
    )
    parser.add_argument(
        "--prompt-stage",
        choices=("stage1", "stage2"),
        default="stage1",
        help="Use the declared stage1 or stage2 prompt for every selected source.",
    )
    parser.add_argument(
        "--source-id",
        action="append",
        dest="source_ids",
        default=[],
        help=(
            "Run only this source ID from the selected input; repeat for a reviewed subset."
        ),
    )
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
    parser.add_argument(
        "--lora-weights",
        type=Path,
        help="Optional local Kontext LoRA file or directory for an exploratory adapted run.",
    )
    parser.add_argument("--lora-scale", type=float, default=1.0)
    args = parser.parse_args()

    if not args.resume and (args.records_output.exists() or args.failures_output.exists()):
        parser.error("refusing to append to an existing records or failures file")
    if args.input_records:
        if args.prompt_stage != "stage2":
            parser.error("--input-records requires --prompt-stage stage2")
        if args.data_root is not None:
            parser.error("--data-root cannot be used with --input-records")
        try:
            manifest_records = load_sequential_inputs(args.input_records)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
    elif args.source_list:
        if args.data_root is None:
            parser.error("--data-root is required with --source-list")
        try:
            pair_bank_rows = load_pair_bank_source_list(
                args.source_list,
                args.data_root,
                roles={"candidate"},
            )
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        manifest_records = [row.as_image_record() for row in pair_bank_rows]
    else:
        if args.data_root is None:
            parser.error("--data-root is required with --manifest")
        manifest_records = load_dataset_manifest(
            args.manifest,
            data_root=args.data_root,
            split=args.split,
        )
    required_styles = tuple(args.required_styles or REQUIRED_STYLES)
    try:
        manifest_records = filter_records_by_source_ids(
            manifest_records, args.source_ids
        )
        selected = select_probe_records(
            manifest_records,
            args.probe_stage,
            required_styles=required_styles,
        )
    except ValueError as exc:
        parser.error(str(exc))
    settings = FluxKontextSettings(
        model_dir=args.model_dir.resolve(),
        download_manifest=args.download_manifest.resolve(),
        hash_manifest=args.hash_manifest.resolve(),
        source_revision=args.source_revision,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        lora_weights=args.lora_weights.resolve() if args.lora_weights else None,
        lora_scale=args.lora_scale,
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
                prompt_stage=args.prompt_stage,
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
    backend = FluxKontextBackend(
        settings,
        styles_config,
        prompt_stage=args.prompt_stage,
    )
    failures = 0
    for record in selected:
        started = time.perf_counter()
        try:
            result = backend.run(record, args.output_dir, seed=args.seed)
            if args.input_records:
                result.extra["sequential_parent_records"] = str(
                    args.input_records.resolve()
                )
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
                    "prompt_stage": args.prompt_stage,
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
