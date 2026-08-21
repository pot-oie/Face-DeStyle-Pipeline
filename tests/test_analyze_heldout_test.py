import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_heldout_test.py"
SPEC = importlib.util.spec_from_file_location("analyze_heldout_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def synthetic_primary() -> pd.DataFrame:
    methods = [
        "prompt_generic",
        "prompt_adaptive",
        "global_canny_0p4",
        "region_canny",
        "flux_kontext_native1024",
    ]
    styles = ("3d_cartoon", "comic", "ink", "watercolor")
    rows = []
    for method_index, method in enumerate(methods):
        for source_index in range(60):
            style = styles[source_index // 15]
            source_id = f"source-{source_index:02d}"
            accepted = source_index < (52 if method_index == 4 else method_index + 2)
            score = 5 if accepted else 3
            rows.append(
                {
                    "canonical_id": f"{method}:{source_id}",
                    "method": method,
                    "source_id": source_id,
                    "style_category": style,
                    "content_score": score,
                    "style_removal_score": score,
                    "identity_score": score,
                    "identity_judgment_valid": "yes",
                    "missing_core": False,
                    "accepted": accepted,
                    "content_dimension_failed": not accepted,
                    "style_dimension_failed": not accepted,
                    "identity_dimension_failed": not accepted,
                    "failure_types": "",
                    "failure_types_reported": False,
                }
            )
    return pd.DataFrame(rows)


def synthetic_evaluations(primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, row in primary.iterrows():
        score = float(row["content_score"]) / 5
        rows.append(
            {
                "method": row["method"],
                "source_id": row["source_id"],
                "style_category": row["style_category"],
                "dinov2_cosine": score,
                "clip_cosine": score,
                "arcface_cosine": None if index == 0 else score,
                "arcface_status": "no_face_generated" if index == 0 else "ok_largest_face",
                "qwen_content_score": None,
                "qwen_style_removal_score": None,
                "qwen_identity_score": None,
            }
        )
    return pd.DataFrame(rows)


def test_frozen_statistical_analysis_on_300_synthetic_pairs() -> None:
    primary = synthetic_primary()
    results = MODULE.analyze(
        primary,
        synthetic_evaluations(primary),
        "flux_kontext_native1024",
    )

    assert len(results["method_pass_rates"]) == 5
    assert len(results["paired_pass"]) == 4
    assert set(results["paired_pass"]["bootstrap_resamples"]) == {20_000}
    assert set(results["paired_pass"]["bootstrap_seed"]) == {20260821}
    assert len(results["ordinal"]) == 12
    assert results["primary_hypothesis_supported"] is True
    assert results["arcface_missing_by_method"]["prompt_generic"] == 1
    assert results["arcface_no_face_by_method"]["prompt_generic"] == 1


def test_exact_mcnemar_wilcoxon_and_missing_arcface_helpers() -> None:
    left = np.array([1, 1, 1, 0])
    right = np.array([0, 0, 1, 0])
    left_only, right_only, p_value = MODULE.exact_mcnemar(left, right)
    assert (left_only, right_only, p_value) == (2, 0, 0.5)

    n, statistic, p_value = MODULE.wilcoxon_exact(
        np.array([5, 4, 3]), np.array([3, 4, 2])
    )
    assert n == 2
    assert statistic == 0
    assert p_value == 0.5
    assert MODULE.roc_auc(np.array([0.1, 0.9]), np.array([0, 1])) == 1.0


def test_single_rater_repeat_agreement() -> None:
    primary = synthetic_primary().iloc[:60].copy()
    repeat = primary.copy()
    result = MODULE.agreement(primary, repeat)

    assert result["repeated_pairs"] == 60
    assert result["content_score"]["quadratic_weighted_cohen_kappa"] == 1.0
    assert result["pass_fail"]["unweighted_cohen_kappa"] == 1.0
