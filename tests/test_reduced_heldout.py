import csv
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from face_destyle.reduced_heldout import METHODS, SCORE_COLUMNS, STYLES, validate_reduced_review

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FREEZE = load_script("freeze_reduced_heldout")
ANALYZE = load_script("analyze_reduced_heldout")


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_review(root: Path) -> Path:
    scores = []
    selection = []
    sources = []
    incomplete = set(range(99, 160))
    row_index = 0
    for style_index, style in enumerate(STYLES):
        for source_index in range(8):
            source_id = f"source-{style_index}-{source_index}"
            complete = 0
            for method_index, method in enumerate(METHODS):
                blind_id = f"B-{row_index:03d}"
                score = 5 if method == "flux_kontext_native1024" else 3 + method_index % 2
                scores.append(
                    {
                        "blind_id": blind_id,
                        "style_category": style,
                        "content_score": score,
                        "style_removal_score": score,
                        "identity_score": score,
                        "identity_judgment_valid": "yes",
                        "failure_types": "" if score >= 4 else "structure_drift",
                        "reviewer_notes": "",
                    }
                )
                was_complete = row_index not in incomplete
                complete += int(was_complete)
                selection.append(
                    {
                        "blind_id": blind_id,
                        "source_id": source_id,
                        "method": method,
                        "style_category": style,
                        "core_complete_at_reduction": str(was_complete),
                    }
                )
                row_index += 1
            sources.append(
                {
                    "style_category": style,
                    "source_id": source_id,
                    "completed_candidates": complete,
                    "remaining_candidates": 5 - complete,
                }
            )
    remaining = [row for index, row in enumerate(scores) if index in incomplete]
    write_csv(root / "selected-160/scores.csv", SCORE_COLUMNS, scores)
    write_csv(root / "remaining-61/scores.csv", SCORE_COLUMNS, remaining)
    write_csv(
        root / "private-unblinded/selection.csv",
        ("blind_id", "source_id", "method", "style_category", "core_complete_at_reduction"),
        selection,
    )
    write_csv(
        root / "private-unblinded/selected-sources.csv",
        ("style_category", "source_id", "completed_candidates", "remaining_candidates"),
        sources,
    )
    (root / "README.md").write_text("synthetic\n", encoding="utf-8")
    (root / "review_app.py").write_text("# synthetic\n", encoding="utf-8")
    return root


def test_validate_freeze_and_analyze_reduced_review(tmp_path: Path) -> None:
    review = build_review(tmp_path / "review")
    validated = validate_reduced_review(review)
    assert validated["summary"]["candidate_count"] == 160
    assert validated["summary"]["source_style_counts"] == {style: 8 for style in STYLES}
    assert validated["summary"]["complete_at_reduction"] == 99

    freeze = FREEZE.build_freeze(review)
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    ANALYZE.verify_freeze(validated, freeze_path)
    results = ANALYZE.analyze_human(pd.DataFrame(validated["rows"]), ANALYZE.FLUX_METHOD)
    assert len(results["method_pass_rates"]) == 5
    assert len(results["paired_pass"]) == 4
    assert set(results["paired_pass"]["complete_source_pairs"]) == {32}
    assert set(results["paired_pass"]["bootstrap_resamples"]) == {20_000}
    assert len(results["ordinal"]) == 12
    assert "not_reported" in set(results["failure_types"]["failure_type"])


def test_reduced_review_rejects_incomplete_method_pairing(tmp_path: Path) -> None:
    review = build_review(tmp_path / "review")
    selection = review / "private-unblinded/selection.csv"
    text = selection.read_text(encoding="utf-8")
    selection.write_text(text.replace("prompt_adaptive", "prompt_generic", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="all five"):
        validate_reduced_review(review)
