#!/usr/bin/env python3
"""Validate, unblind, and summarize two completed human-review rounds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

SCORE_FIELDS = ("content_score", "style_removal_score", "identity_score")
FAILURE_TYPES = {
    "structure_drift",
    "identity_drift",
    "artistic_contour_residual",
    "material_render_residual",
    "background_drift",
    "no_usable_face",
    "other",
}


def load_key(path: Path) -> dict[str, dict[str, str]]:
    rows = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        blind_id = str(row["blind_id"])
        if blind_id in rows:
            raise ValueError(f"duplicate blind ID in key at line {line_number}: {blind_id}")
        rows[blind_id] = row
    return rows


def parse_score(value: str, field: str, blind_id: str, *, allow_blank: bool = False) -> int | None:
    value = value.strip()
    if allow_blank and not value:
        return None
    if value not in {"0", "1", "2", "3", "4", "5"}:
        raise ValueError(f"{blind_id}: {field} must be an integer from 0 to 5")
    return int(value)


def load_round(path: Path, key: dict[str, dict[str, str]], expected_round: str) -> list[dict]:
    output = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            blind_id = row["blind_id"].strip()
            if blind_id not in key:
                raise ValueError(f"unknown blind ID in {path}: {blind_id}")
            identity_valid = row["identity_judgment_valid"].strip().lower()
            if identity_valid not in {"yes", "no"}:
                raise ValueError(f"{blind_id}: identity_judgment_valid must be yes or no")
            content = parse_score(row["content_score"], "content_score", blind_id)
            style = parse_score(row["style_removal_score"], "style_removal_score", blind_id)
            identity = parse_score(
                row["identity_score"],
                "identity_score",
                blind_id,
                allow_blank=identity_valid == "no",
            )
            failures = [
                item.strip()
                for item in row["failure_types"].split(";")
                if item.strip()
            ]
            unknown = sorted(set(failures) - FAILURE_TYPES)
            if unknown:
                raise ValueError(f"{blind_id}: unknown failure types: {', '.join(unknown)}")
            private = key[blind_id]
            if private["round"] != expected_round:
                raise ValueError(f"{blind_id}: key round does not match {expected_round}")
            accepted = bool(
                content >= 4
                and style >= 4
                and (identity_valid == "no" or (identity is not None and identity >= 4))
            )
            output.append(
                {
                    "round": expected_round,
                    "blind_id": blind_id,
                    "canonical_id": private["canonical_id"],
                    "method": private["method"],
                    "source_id": private["source_id"],
                    "style_category": private["style_category"],
                    "content_score": content,
                    "style_removal_score": style,
                    "identity_score": identity,
                    "identity_judgment_valid": identity_valid,
                    "accepted": accepted,
                    "failure_types": ";".join(failures),
                    "reviewer_notes": row["reviewer_notes"],
                }
            )
    return output


def agreement(rows: list[dict]) -> dict:
    paired: defaultdict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        paired[row["canonical_id"]][row["round"]] = row
    complete = [value for value in paired.values() if set(value) == {"a", "b"}]
    result = {"paired_candidates": len(complete)}
    for field in SCORE_FIELDS:
        pairs = [
            (value["a"][field], value["b"][field])
            for value in complete
            if value["a"][field] is not None and value["b"][field] is not None
        ]
        result[field] = {
            "n": len(pairs),
            "exact_agreement_rate": (
                sum(left == right for left, right in pairs) / len(pairs) if pairs else None
            ),
            "mean_absolute_difference": (
                sum(abs(left - right) for left, right in pairs) / len(pairs)
                if pairs
                else None
            ),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-a", type=Path, required=True)
    parser.add_argument("--round-b", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    key = load_key(args.private_key)
    rows = load_round(args.round_a, key, "a") + load_round(args.round_b, key, "b")
    if len(rows) != len(key):
        raise ValueError(f"expected {len(key)} scored rows from key, received {len(rows)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "unblinded-human-scores.csv", index=False)

    metric_columns = [*SCORE_FIELDS, "accepted"]
    by_method = frame.groupby("method")[metric_columns].agg(["count", "mean"])
    by_method.to_csv(args.output_dir / "human-summary-by-method.csv")
    by_method_style = frame.groupby(["method", "style_category"])[metric_columns].agg(
        ["count", "mean"]
    )
    by_method_style.to_csv(args.output_dir / "human-summary-by-method-style.csv")

    failure_counts: Counter[str] = Counter()
    for value in frame["failure_types"]:
        failure_counts.update(item for item in value.split(";") if item)
    summary = {
        "scored_rows": len(rows),
        "unique_candidates": frame["canonical_id"].nunique(),
        "acceptance_rule": (
            "content>=4 and style_removal>=4 and "
            "(identity>=4 when identity_judgment_valid=yes)"
        ),
        "failure_counts": dict(sorted(failure_counts.items())),
        "round_agreement": agreement(rows),
        "note": "Pilot human review; not held-out test evidence.",
    }
    (args.output_dir / "human-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Validated and summarized {len(rows)} blinded scores in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
