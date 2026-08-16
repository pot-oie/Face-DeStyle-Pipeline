#!/usr/bin/env python3
"""Run checkpointed DINO, CLIP, ArcFace, and Qwen pair evaluation from local assets."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path
from typing import Any

from tqdm import tqdm

from face_destyle.data.metadata import read_jsonl
from face_destyle.metrics.content import ClipPairMetric, DinoV2PairMetric
from face_destyle.metrics.formal import apply_scalar_metric, build_formal_records, checkpoint
from face_destyle.metrics.identity import ArcFacePairMetric
from face_destyle.metrics.style_removal import RUBRIC_VERSION, QwenPairRubric
from face_destyle.models import ModelRegistry
from face_destyle.schemas import FormalEvaluationRecord

METRICS = ("dino", "clip", "arcface", "qwen")


def _metadata(asset: str, location: Path) -> dict[str, Any]:
    return {"asset": asset, "resolved_model_path": str(location), "host": platform.node()}


def _load_or_create(args: argparse.Namespace) -> list[FormalEvaluationRecord]:
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f"output exists; pass --resume to continue: {args.output}")
        return read_jsonl(args.output, FormalEvaluationRecord)
    records = build_formal_records(args.records)
    checkpoint(records, args.output)
    return records


def _run_arcface(
    records: list[FormalEvaluationRecord],
    metric: ArcFacePairMetric,
    output: Path,
    metadata: dict[str, Any],
    retry_failures: bool,
) -> None:
    for record in tqdm(records, desc="arcface"):
        if record.arcface_status is not None:
            continue
        if "arcface" in record.failures and not retry_failures:
            continue
        try:
            score, status = metric.score(record.input_path, record.output_path)
            record.arcface_cosine = score
            record.arcface_status = status
            record.failures.pop("arcface", None)
            record.evaluator_metadata["arcface"] = metadata
        except Exception as exc:
            record.failures["arcface"] = f"{type(exc).__name__}: {exc}"
        checkpoint(records, output)


def _run_qwen(
    records: list[FormalEvaluationRecord],
    metric: QwenPairRubric,
    output: Path,
    metadata: dict[str, Any],
    retry_failures: bool,
) -> None:
    for record in tqdm(records, desc="qwen"):
        if record.qwen_style_removal_score is not None:
            continue
        if "qwen" in record.failures and not retry_failures:
            continue
        try:
            payload, raw = metric.score(
                record.input_path, record.output_path, record.style_category
            )
            record.qwen_content_score = payload["content_preservation"]
            record.qwen_style_removal_score = payload["style_removal"]
            record.qwen_identity_score = payload["identity_preservation"]
            record.qwen_evidence = payload["evidence"]
            record.qwen_raw_response = raw
            record.failures.pop("qwen", None)
            record.evaluator_metadata["qwen"] = {
                **metadata,
                "rubric_version": RUBRIC_VERSION,
            }
        except Exception as exc:
            record.failures["qwen"] = f"{type(exc).__name__}: {exc}"
        checkpoint(records, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        action="append",
        required=True,
        metavar="METHOD=PATH",
        help="Method label and DestylizationRecord JSONL; repeat for every run.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--metric", action="append", choices=METRICS)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--arcface-provider", choices=("auto", "cuda", "cpu"), default="auto"
    )
    parser.add_argument("--qwen-max-new-tokens", type=int, default=160)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()
    if args.qwen_max_new_tokens < 32:
        parser.error("--qwen-max-new-tokens must be at least 32")

    selected = args.metric or list(METRICS)
    registry = ModelRegistry.from_yaml(args.models_config)
    asset_names = {
        "dino": "dinov2_base",
        "clip": "clip_vit_l14",
        "arcface": "insightface_buffalo_l",
        "qwen": "qwen25_vl_3b",
    }
    resolved: dict[str, Path] = {}
    for name in selected:
        asset = asset_names[name]
        check = registry.check(asset)
        if not check.available or check.location is None:
            raise FileNotFoundError(f"required asset is unavailable: {asset}: {check}")
        resolved[name] = check.location

    records = _load_or_create(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if "dino" in selected:
        metric = DinoV2PairMetric(resolved["dino"], device=args.device)
        apply_scalar_metric(
            records,
            metric_name="dino",
            score_field="dinov2_cosine",
            scorer=metric.score,
            output=args.output,
            retry_failures=args.retry_failures,
        )
        for record in records:
            if record.dinov2_cosine is not None:
                record.evaluator_metadata["dino"] = _metadata(
                    "dinov2_base", resolved["dino"]
                )
        checkpoint(records, args.output)
        metric.close()

    if "clip" in selected:
        metric = ClipPairMetric(resolved["clip"], device=args.device)
        apply_scalar_metric(
            records,
            metric_name="clip",
            score_field="clip_cosine",
            scorer=metric.score,
            output=args.output,
            retry_failures=args.retry_failures,
        )
        for record in records:
            if record.clip_cosine is not None:
                record.evaluator_metadata["clip"] = _metadata(
                    "clip_vit_l14", resolved["clip"]
                )
        checkpoint(records, args.output)
        metric.close()

    if "arcface" in selected:
        metric = ArcFacePairMetric(
            resolved["arcface"], provider=args.arcface_provider
        )
        _run_arcface(
            records,
            metric,
            args.output,
            _metadata("insightface_buffalo_l", resolved["arcface"]),
            args.retry_failures,
        )
        del metric

    if "qwen" in selected:
        metric = QwenPairRubric(
            resolved["qwen"],
            device=args.device,
            max_new_tokens=args.qwen_max_new_tokens,
        )
        _run_qwen(
            records,
            metric,
            args.output,
            _metadata("qwen25_vl_3b", resolved["qwen"]),
            args.retry_failures,
        )
        metric.close()
        del metric

    failures = sum(bool(record.failures) for record in records)
    print(f"Wrote {len(records)} raw formal evaluations to {args.output}; failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
