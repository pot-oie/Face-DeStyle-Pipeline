#!/usr/bin/env python3
"""Plan and resolve a lightweight human-reviewed multi-style processing route."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from face_destyle.routing import (
    REVIEW_FIELDS,
    build_review_rows,
    build_stage1_plan,
    load_records,
    load_review,
    resolve_review,
    write_csv,
    write_jsonl,
)


def refuse_existing(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise ValueError(f"refusing to overwrite existing route artifacts: {existing}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan-stage1")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--styles-config", type=Path, default=Path("configs/styles.yaml"))
    plan.add_argument("--output", type=Path, required=True)

    review = subparsers.add_parser("init-review")
    review.add_argument("--records", type=Path, required=True)
    review.add_argument(
        "--current-stage", choices=("stage1", "stage2", "origami_v1"), required=True
    )
    review.add_argument("--output", type=Path, required=True)

    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--review", type=Path, required=True)
    resolve.add_argument("--records", type=Path, required=True)
    resolve.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "plan-stage1":
            refuse_existing([args.output])
            styles = yaml.safe_load(args.styles_config.read_text(encoding="utf-8"))
            rows = build_stage1_plan(args.manifest, styles)
            fields = ("source_id", "style_category", "asset_path", "planned_stage", "prompt")
            write_csv(args.output, rows, fields)
            print(f"Planned Stage 1 for {len(rows)} sources -> {args.output}")
            return 0

        if args.command == "init-review":
            refuse_existing([args.output])
            records = load_records(args.records)
            write_csv(args.output, build_review_rows(records, args.current_stage), REVIEW_FIELDS)
            print(f"Initialized {len(records)} review rows -> {args.output}")
            return 0

        routes_path = args.output_dir / "routes.jsonl"
        stage2_path = args.output_dir / "stage2-input-records.jsonl"
        origami_path = args.output_dir / "origami-v1-source-ids.txt"
        refuse_existing([routes_path, stage2_path, origami_path])
        rows = load_review(args.review)
        records = load_records(args.records)
        routes, stage2_inputs, origami_ids = resolve_review(
            rows, records, records_path=args.records
        )
        write_jsonl(routes_path, routes)
        if stage2_inputs:
            write_jsonl(stage2_path, stage2_inputs)
        if origami_ids:
            origami_path.parent.mkdir(parents=True, exist_ok=True)
            origami_path.write_text("\n".join(origami_ids) + "\n", encoding="utf-8")
        counts: dict[str, int] = {}
        for route in routes:
            counts[route["status"]] = counts.get(route["status"], 0) + 1
        print(f"Resolved {len(routes)} routes -> {routes_path}; status_counts={counts}")
        return 0
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
