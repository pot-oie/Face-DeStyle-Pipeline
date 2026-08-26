"""Human-reviewed routing helpers for multi-style destylization runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from face_destyle.data.metadata import read_jsonl
from face_destyle.schemas import DatasetManifestRecord, DestylizationRecord

REVIEW_FIELDS = (
    "source_id",
    "style_category",
    "current_stage",
    "current_output_path",
    "decision",
    "reviewer_notes",
)
DECISIONS = {
    "accept_stage1",
    "accept_stage2",
    "accept_origami_v1",
    "run_stage2",
    "run_origami_v1",
    "route_teacher",
    "explicit_failure",
}
ACCEPT_DECISIONS = {
    "accept_stage1": "stage1",
    "accept_stage2": "stage2",
    "accept_origami_v1": "origami_v1",
}


def load_records(path: Path) -> list[DestylizationRecord]:
    records = read_jsonl(path, DestylizationRecord)
    source_ids: set[str] = set()
    for record in records:
        if record.source_id in source_ids:
            raise ValueError(f"duplicate source_id in records: {record.source_id}")
        source_ids.add(record.source_id)
    if not records:
        raise ValueError("records file contains no successful outputs")
    return records


def build_stage1_plan(manifest: Path, styles_config: dict[str, Any]) -> list[dict[str, str]]:
    records = read_jsonl(manifest, DatasetManifestRecord)
    styles = styles_config.get("styles", {})
    plan: list[dict[str, str]] = []
    for record in records:
        style = styles.get(record.style_category, {})
        prompt = style.get("stage1_prompt")
        if not prompt:
            raise ValueError(f"missing Stage 1 prompt for style {record.style_category!r}")
        plan.append(
            {
                "source_id": record.source_id,
                "style_category": record.style_category,
                "asset_path": str(record.asset_path),
                "planned_stage": "stage1",
                "prompt": str(prompt),
            }
        )
    return plan


def build_review_rows(
    records: list[DestylizationRecord], current_stage: str
) -> list[dict[str, str]]:
    return [
        {
            "source_id": record.source_id,
            "style_category": record.style_category,
            "current_stage": current_stage,
            "current_output_path": str(record.output_path),
            "decision": "",
            "reviewer_notes": "",
        }
        for record in records
    ]


def write_csv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_review(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REVIEW_FIELDS:
            raise ValueError(f"review CSV fields must be {REVIEW_FIELDS}")
        rows = list(reader)
    if not rows:
        raise ValueError("review CSV contains no rows")
    source_ids: set[str] = set()
    for row in rows:
        source_id = row["source_id"].strip()
        decision = row["decision"].strip()
        if not source_id or source_id in source_ids:
            raise ValueError(f"missing or duplicate source_id in review: {source_id!r}")
        if decision not in DECISIONS:
            raise ValueError(f"invalid or missing decision for {source_id}: {decision!r}")
        source_ids.add(source_id)
    return rows


def resolve_review(
    rows: list[dict[str, str]],
    records: list[DestylizationRecord],
    *,
    records_path: Path,
) -> tuple[list[dict[str, Any]], list[DestylizationRecord], list[str]]:
    by_source = {record.source_id: record for record in records}
    review_ids = {row["source_id"] for row in rows}
    if review_ids != set(by_source):
        raise ValueError(
            "review and records source IDs differ: "
            f"review_only={sorted(review_ids - set(by_source))}, "
            f"records_only={sorted(set(by_source) - review_ids)}"
        )

    routes: list[dict[str, Any]] = []
    stage2_inputs: list[DestylizationRecord] = []
    origami_v1_ids: list[str] = []
    for row in rows:
        record = by_source[row["source_id"]]
        current_stage = row["current_stage"].strip()
        decision = row["decision"].strip()
        if row["style_category"].strip() != record.style_category:
            raise ValueError(f"style mismatch for {record.source_id}")
        if Path(row["current_output_path"].strip()) != record.output_path:
            raise ValueError(f"current output mismatch for {record.source_id}")
        status = "terminal_success"
        terminal_route: str | None = decision
        terminal_output: str | None = str(record.output_path)
        if decision in ACCEPT_DECISIONS:
            expected = ACCEPT_DECISIONS[decision]
            if current_stage != expected:
                raise ValueError(
                    f"{decision} requires current_stage={expected} for {record.source_id}"
                )
        elif decision == "run_stage2":
            if current_stage not in {"stage1", "origami_v1"}:
                raise ValueError(f"run_stage2 cannot follow {current_stage!r}")
            status = "action_required"
            terminal_route = None
            terminal_output = None
            stage2_inputs.append(record)
        elif decision == "run_origami_v1":
            if record.style_category != "origami" or current_stage != "stage1":
                raise ValueError("run_origami_v1 requires an Origami Stage 1 record")
            status = "action_required"
            terminal_route = None
            terminal_output = None
            origami_v1_ids.append(record.source_id)
        elif decision == "route_teacher":
            status = "external_pending"
            terminal_output = None
        elif decision == "explicit_failure":
            status = "terminal_failure"
            terminal_output = None
        routes.append(
            {
                "source_id": record.source_id,
                "style_category": record.style_category,
                "current_stage": current_stage,
                "decision": decision,
                "status": status,
                "parent_records": str(records_path.resolve()),
                "current_output_path": str(record.output_path),
                "terminal_route": terminal_route,
                "terminal_output_path": terminal_output,
                "reviewer_notes": row["reviewer_notes"].strip(),
            }
        )
    return routes, stage2_inputs, origami_v1_ids


def write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = row.model_dump(mode="json") if hasattr(row, "model_dump") else row
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
