"""Checkpointed orchestration helpers for raw formal pair metrics."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from face_destyle.data.metadata import read_jsonl, write_jsonl
from face_destyle.schemas import DestylizationRecord, FormalEvaluationRecord


def parse_record_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError("record input must use METHOD=/path/to/records.jsonl")
    method, raw_path = spec.split("=", 1)
    if not method.strip() or not raw_path.strip():
        raise ValueError("record input must use METHOD=/path/to/records.jsonl")
    return method.strip(), Path(raw_path).expanduser().resolve()


def build_formal_records(specs: list[str]) -> list[FormalEvaluationRecord]:
    evaluations: list[FormalEvaluationRecord] = []
    seen: set[str] = set()
    for spec in specs:
        method, path = parse_record_spec(spec)
        records = read_jsonl(
            path,
            DestylizationRecord,
            check_paths=True,
            path_fields=("input_path", "output_path"),
        )
        for record in records:
            evaluation_id = f"{method}:{record.id}"
            if evaluation_id in seen:
                raise ValueError(f"duplicate formal evaluation id: {evaluation_id}")
            seen.add(evaluation_id)
            evaluations.append(
                FormalEvaluationRecord(
                    id=evaluation_id,
                    record_id=record.id,
                    source_id=record.source_id,
                    method=method,
                    input_path=record.input_path,
                    output_path=record.output_path,
                    style_category=record.style_category,
                )
            )
    return evaluations


def checkpoint(records: list[FormalEvaluationRecord], output: Path) -> None:
    temporary = output.with_suffix(output.suffix + ".tmp")
    write_jsonl(records, temporary)
    temporary.replace(output)


def apply_scalar_metric(
    records: list[FormalEvaluationRecord],
    *,
    metric_name: str,
    score_field: str,
    scorer: Callable[[Path, Path], float],
    output: Path,
    retry_failures: bool = False,
) -> None:
    for record in records:
        if getattr(record, score_field) is not None:
            continue
        if metric_name in record.failures and not retry_failures:
            continue
        try:
            setattr(record, score_field, scorer(record.input_path, record.output_path))
            record.failures.pop(metric_name, None)
        except Exception as exc:
            record.failures[metric_name] = f"{type(exc).__name__}: {exc}"
        checkpoint(records, output)
