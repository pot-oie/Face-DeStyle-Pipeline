"""Validation helpers for the operator-amended reduced held-out review."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

METHODS = (
    "flux_kontext_native1024",
    "global_canny_0p4",
    "prompt_adaptive",
    "prompt_generic",
    "region_canny",
)
STYLES = ("3d_cartoon", "comic", "ink", "watercolor")
FAILURE_TYPES = {
    "structure_drift",
    "identity_drift",
    "artistic_contour_residual",
    "material_render_residual",
    "background_drift",
    "no_usable_face",
    "other",
}
SCORE_COLUMNS = (
    "blind_id",
    "style_category",
    "content_score",
    "style_removal_score",
    "identity_score",
    "identity_judgment_valid",
    "failure_types",
    "reviewer_notes",
)
SELECTION_COLUMNS = (
    "blind_id",
    "source_id",
    "method",
    "style_category",
    "core_complete_at_reduction",
)
SOURCE_COLUMNS = (
    "style_category",
    "source_id",
    "completed_candidates",
    "remaining_candidates",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(
                f"{path} columns differ: expected {columns}, received {reader.fieldnames}"
            )
        return list(reader)


def _unique_by(rows: list[dict[str, str]], field: str, label: str) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row[field].strip()
        if not value or value in output:
            raise ValueError(f"{label} contains blank or duplicate {field}: {value!r}")
        output[value] = row
    return output


def _parse_complete(value: str, blind_id: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{blind_id}: core_complete_at_reduction must be True or False")


def validate_reduced_review(review_dir: Path) -> dict[str, Any]:
    """Validate the exact 32-source/160-candidate amended review without reading images."""
    root = review_dir.expanduser().resolve()
    paths = {
        "readme": root / "README.md",
        "selected_scores": root / "selected-160" / "scores.csv",
        "remaining_scores": root / "remaining-61" / "scores.csv",
        "selection": root / "private-unblinded" / "selection.csv",
        "selected_sources": root / "private-unblinded" / "selected-sources.csv",
        "review_app": root / "review_app.py",
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing reduced-review {label}: {path}")

    scores = _read_csv(paths["selected_scores"], SCORE_COLUMNS)
    remaining = _read_csv(paths["remaining_scores"], SCORE_COLUMNS)
    selection = _read_csv(paths["selection"], SELECTION_COLUMNS)
    sources = _read_csv(paths["selected_sources"], SOURCE_COLUMNS)
    if len(scores) != 160 or len(selection) != 160:
        raise ValueError("selected scores and selection must each contain exactly 160 rows")
    if len(remaining) != 61 or len(sources) != 32:
        raise ValueError("remaining scores must have 61 rows and selected sources 32 rows")

    scores_by_id = _unique_by(scores, "blind_id", "selected scores")
    remaining_by_id = _unique_by(remaining, "blind_id", "remaining scores")
    selection_by_id = _unique_by(selection, "blind_id", "selection")
    sources_by_id = _unique_by(sources, "source_id", "selected sources")
    if set(scores_by_id) != set(selection_by_id):
        raise ValueError("selection does not exactly explain the 160 selected score rows")

    core_complete: dict[str, bool] = {}
    source_methods: defaultdict[str, set[str]] = defaultdict(set)
    source_styles: dict[str, str] = {}
    normalized: list[dict[str, Any]] = []
    for blind_id, score_row in scores_by_id.items():
        selected = selection_by_id[blind_id]
        style = score_row["style_category"].strip()
        if selected["style_category"].strip() != style or style not in STYLES:
            raise ValueError(f"{blind_id}: score/selection style mismatch or unknown style")
        method = selected["method"].strip()
        if method not in METHODS:
            raise ValueError(f"{blind_id}: unknown method {method!r}")
        source_id = selected["source_id"].strip()
        if not source_id:
            raise ValueError(f"{blind_id}: blank source_id")
        source_methods[source_id].add(method)
        previous_style = source_styles.setdefault(source_id, style)
        if previous_style != style:
            raise ValueError(f"{source_id}: crosses styles in selection")
        core_complete[blind_id] = _parse_complete(selected["core_complete_at_reduction"], blind_id)

        parsed_scores: dict[str, int] = {}
        for field in ("content_score", "style_removal_score", "identity_score"):
            value = score_row[field].strip()
            if value not in {"0", "1", "2", "3", "4", "5"}:
                raise ValueError(f"{blind_id}: {field} must be a complete integer from 0 to 5")
            parsed_scores[field] = int(value)
        identity_valid = score_row["identity_judgment_valid"].strip().lower()
        if identity_valid not in {"yes", "no"}:
            raise ValueError(f"{blind_id}: identity_judgment_valid must be yes or no")
        failures = [item.strip() for item in score_row["failure_types"].split(";") if item.strip()]
        unknown = sorted(set(failures) - FAILURE_TYPES)
        if unknown:
            raise ValueError(f"{blind_id}: unknown failure types: {', '.join(unknown)}")
        accepted = bool(
            parsed_scores["content_score"] >= 4
            and parsed_scores["style_removal_score"] >= 4
            and identity_valid == "yes"
            and parsed_scores["identity_score"] >= 4
        )
        normalized.append(
            {
                "blind_id": blind_id,
                "canonical_id": f"{method}:{source_id}",
                "source_id": source_id,
                "method": method,
                "style_category": style,
                **parsed_scores,
                "identity_judgment_valid": identity_valid,
                "accepted": accepted,
                "missing_core": False,
                "content_dimension_failed": parsed_scores["content_score"] < 4,
                "style_dimension_failed": parsed_scores["style_removal_score"] < 4,
                "identity_dimension_failed": (
                    identity_valid != "yes" or parsed_scores["identity_score"] < 4
                ),
                "failure_types": ";".join(failures),
                "failure_types_reported": bool(failures),
                "core_complete_at_reduction": core_complete[blind_id],
            }
        )

    if len(source_methods) != 32 or set(source_methods) != set(sources_by_id):
        raise ValueError("selected source list does not exactly explain the 32 selection sources")
    expected_methods = set(METHODS)
    if any(methods != expected_methods for methods in source_methods.values()):
        raise ValueError("every selected source must contain exactly all five frozen methods")
    if Counter(source_styles.values()) != Counter({style: 8 for style in STYLES}):
        raise ValueError("selected sources must contain exactly eight sources per style")
    if Counter(row["style_category"] for row in normalized) != Counter(
        {style: 40 for style in STYLES}
    ):
        raise ValueError("selected candidates must contain exactly 40 rows per style")
    if Counter(row["method"] for row in normalized) != Counter({method: 32 for method in METHODS}):
        raise ValueError("selected candidates must contain exactly 32 rows per method")

    false_ids = {blind_id for blind_id, complete in core_complete.items() if not complete}
    if false_ids != set(remaining_by_id):
        raise ValueError("remaining-61 must exactly match rows incomplete at reduction")
    for blind_id, row in remaining_by_id.items():
        if row != scores_by_id[blind_id]:
            raise ValueError(f"{blind_id}: remaining score does not match selected score")
    if sum(core_complete.values()) != 99:
        raise ValueError("exactly 99 selected candidates must have been complete at reduction")

    for source_id, source_row in sources_by_id.items():
        style = source_row["style_category"].strip()
        if style != source_styles[source_id]:
            raise ValueError(f"{source_id}: selected-source style mismatch")
        selected_rows = [row for row in selection if row["source_id"] == source_id]
        complete_count = sum(
            _parse_complete(row["core_complete_at_reduction"], row["blind_id"])
            for row in selected_rows
        )
        if source_row["completed_candidates"] != str(complete_count) or source_row[
            "remaining_candidates"
        ] != str(5 - complete_count):
            raise ValueError(f"{source_id}: selected-source completion counts do not explain rows")

    input_sha256 = {label: file_sha256(path) for label, path in paths.items()}
    summary = {
        "candidate_count": 160,
        "source_count": 32,
        "method_count": 5,
        "methods": list(METHODS),
        "source_style_counts": dict(sorted(Counter(source_styles.values()).items())),
        "candidate_style_counts": dict(
            sorted(Counter(row["style_category"] for row in normalized).items())
        ),
        "method_counts": dict(sorted(Counter(row["method"] for row in normalized).items())),
        "complete_at_reduction": 99,
        "completed_after_reduction": 61,
        "identity_valid_counts": dict(
            sorted(Counter(row["identity_judgment_valid"] for row in normalized).items())
        ),
        "failure_types_not_reported": sum(not row["failure_types_reported"] for row in normalized),
    }
    return {
        "root": root,
        "paths": paths,
        "input_sha256": input_sha256,
        "rows": normalized,
        "selection": selection,
        "selected_sources": sources,
        "summary": summary,
    }


def validate_private_key_mapping(
    private_key_path: Path, selection: list[dict[str, str]]
) -> dict[str, int]:
    """Verify the reduced unblinding fields against the sealed original key."""
    key: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        private_key_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        blind_id = str(row.get("blind_id", ""))
        if not blind_id or blind_id in key:
            raise ValueError(f"private key line {line_number} has blank or duplicate blind_id")
        key[blind_id] = row
    for selected in selection:
        blind_id = selected["blind_id"]
        private = key.get(blind_id)
        if private is None:
            raise ValueError(f"{blind_id}: missing from private key")
        if private.get("round") != "primary":
            raise ValueError(f"{blind_id}: selected candidate is not from the primary round")
        for field in ("source_id", "method", "style_category"):
            if str(private.get(field, "")) != selected[field]:
                raise ValueError(f"{blind_id}: private-key {field} does not match selection")
    return {
        "private_key_rows": len(key),
        "selected_rows_verified": len(selection),
        "mapping_errors": 0,
    }
