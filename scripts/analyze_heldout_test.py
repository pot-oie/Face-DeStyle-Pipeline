#!/usr/bin/env python3
"""Unblind and run the frozen formal-v1 held-out statistical analysis once."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from face_destyle.data.metadata import read_jsonl
from face_destyle.schemas import FormalEvaluationRecord

BOOTSTRAP_RESAMPLES = 20_000
ANALYSIS_SEED = 20260821
SCORE_FIELDS = ("content_score", "style_removal_score", "identity_score")
AUTO_METRICS = (
    "dinov2_cosine",
    "clip_cosine",
    "arcface_cosine",
    "qwen_content_score",
    "qwen_style_removal_score",
    "qwen_identity_score",
)
FAILURE_TYPES = {
    "structure_drift",
    "identity_drift",
    "artistic_contour_residual",
    "material_render_residual",
    "background_drift",
    "no_usable_face",
    "other",
}


def average_ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def spearman(values: np.ndarray, targets: np.ndarray) -> float | None:
    if len(values) < 2:
        return None
    left = average_ranks(values)
    right = average_ranks(targets)
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def roc_auc(values: np.ndarray, labels: np.ndarray) -> float | None:
    positive = labels == 1
    n_positive = int(positive.sum())
    n_negative = len(labels) - n_positive
    if not n_positive or not n_negative:
        return None
    ranks = average_ranks(values)
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2) / (
        n_positive * n_negative
    )


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (math.nan, math.nan)
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    ) / denominator
    return center - half, center + half


def exact_mcnemar(left: np.ndarray, right: np.ndarray) -> tuple[int, int, float]:
    left_wins = int(np.sum((left == 1) & (right == 0)))
    right_wins = int(np.sum((left == 0) & (right == 1)))
    discordant = left_wins + right_wins
    if discordant == 0:
        return left_wins, right_wins, 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(left_wins, right_wins) + 1))
    return left_wins, right_wins, min(1.0, 2 * tail / (2**discordant))


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    ordered = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(ordered):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def bootstrap_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = ANALYSIS_SEED,
) -> tuple[float, float, float]:
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    if not len(differences):
        return math.nan, math.nan, math.nan
    indices = np.random.default_rng(seed).integers(
        0, len(differences), size=(resamples, len(differences))
    )
    replicates = differences[indices].mean(axis=1)
    low, high = np.quantile(replicates, [0.025, 0.975])
    return float(differences.mean()), float(low), float(high)


def wilcoxon_exact(left: np.ndarray, right: np.ndarray) -> tuple[int, float, float]:
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    differences = differences[differences != 0]
    if not len(differences):
        return 0, 0.0, 1.0
    scaled_ranks = np.rint(average_ranks(np.abs(differences)) * 2).astype(int)
    observed = int(scaled_ranks[differences > 0].sum())
    total_rank = int(scaled_ranks.sum())
    counts = [0] * (total_rank + 1)
    counts[0] = 1
    reachable = 0
    for rank in scaled_ranks:
        for subtotal in range(reachable, -1, -1):
            if counts[subtotal]:
                counts[subtotal + rank] += counts[subtotal]
        reachable += int(rank)
    lower = min(observed, total_rank - observed)
    probability = min(1.0, 2 * sum(counts[: lower + 1]) / (2 ** len(scaled_ranks)))
    statistic = min(observed, total_rank - observed) / 2
    return len(differences), float(statistic), float(probability)


def cohen_kappa(left: np.ndarray, right: np.ndarray, *, quadratic: bool) -> float | None:
    if not len(left):
        return None
    labels = sorted(set(left.tolist()) | set(right.tolist()))
    if len(labels) == 1:
        return 1.0
    positions = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    for first, second in zip(left, right, strict=True):
        matrix[positions[first], positions[second]] += 1
    observed = matrix / matrix.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    if quadratic:
        scale = max(1, len(labels) - 1)
        weights = np.fromfunction(
            lambda i, j: ((i - j) / scale) ** 2,
            (len(labels), len(labels)),
        )
    else:
        weights = np.ones_like(matrix) - np.eye(len(labels))
    observed_disagreement = float((weights * observed).sum())
    expected_disagreement = float((weights * expected).sum())
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else None
    return 1 - observed_disagreement / expected_disagreement


def parse_optional_score(value: str, field: str, blind_id: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    if value not in {"0", "1", "2", "3", "4", "5"}:
        raise ValueError(f"{blind_id}: {field} must be blank or an integer from 0 to 5")
    return int(value)


def load_key(path: Path) -> dict[str, dict]:
    rows = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        blind_id = str(row["blind_id"])
        if blind_id in rows:
            raise ValueError(f"duplicate blind ID in key line {line_number}: {blind_id}")
        rows[blind_id] = row
    return rows


def load_scores(path: Path, key: dict[str, dict], round_name: str) -> list[dict]:
    output = []
    seen = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            blind_id = row["blind_id"].strip()
            if blind_id in seen or blind_id not in key:
                raise ValueError(f"duplicate or unknown blind ID in {path}: {blind_id}")
            seen.add(blind_id)
            private = key[blind_id]
            if private["round"] != round_name:
                raise ValueError(f"{blind_id}: key round mismatch")
            if row["style_category"].strip() != private["style_category"]:
                raise ValueError(f"{blind_id}: style category mismatch")
            scores = {
                field: parse_optional_score(row[field], field, blind_id)
                for field in SCORE_FIELDS
            }
            identity_valid = row["identity_judgment_valid"].strip().lower()
            if identity_valid not in {"", "yes", "no"}:
                raise ValueError(f"{blind_id}: identity_judgment_valid must be yes, no, or blank")
            failures = [item.strip() for item in row["failure_types"].split(";") if item.strip()]
            unknown = sorted(set(failures) - FAILURE_TYPES)
            if unknown:
                raise ValueError(f"{blind_id}: unknown failure types: {', '.join(unknown)}")
            missing_core = (
                scores["content_score"] is None
                or scores["style_removal_score"] is None
                or identity_valid == ""
                or (identity_valid == "yes" and scores["identity_score"] is None)
            )
            accepted = bool(
                not missing_core
                and scores["content_score"] >= 4
                and scores["style_removal_score"] >= 4
                and identity_valid == "yes"
                and scores["identity_score"] is not None
                and scores["identity_score"] >= 4
            )
            output.append(
                {
                    "round": round_name,
                    "blind_id": blind_id,
                    "canonical_id": private["canonical_id"],
                    "method": private["method"],
                    "source_id": private["source_id"],
                    "style_category": private["style_category"],
                    **scores,
                    "identity_judgment_valid": identity_valid or "missing",
                    "missing_core": missing_core,
                    "accepted": accepted,
                    "content_dimension_failed": (
                        scores["content_score"] is None or scores["content_score"] < 4
                    ),
                    "style_dimension_failed": (
                        scores["style_removal_score"] is None
                        or scores["style_removal_score"] < 4
                    ),
                    "identity_dimension_failed": (
                        identity_valid != "yes"
                        or scores["identity_score"] is None
                        or scores["identity_score"] < 4
                    ),
                    "failure_types": ";".join(failures),
                    "failure_types_reported": bool(failures),
                }
            )
    expected = {blind_id for blind_id, row in key.items() if row["round"] == round_name}
    if seen != expected:
        raise ValueError(f"{round_name} score file does not exactly cover its frozen blind IDs")
    return output


def validate_primary(frame: pd.DataFrame, flux_method: str) -> list[str]:
    if len(frame) != 300 or frame["canonical_id"].nunique() != 300:
        raise ValueError("primary round must contain exactly 300 unique method-source candidates")
    methods = sorted(frame["method"].unique().tolist())
    if len(methods) != 5 or flux_method not in methods:
        raise ValueError("primary round must contain FLUX plus exactly four baselines")
    cell_counts = frame.groupby(["method", "style_category"]).size()
    if len(cell_counts) != 20 or set(cell_counts.tolist()) != {15}:
        raise ValueError("primary method-style cells must each contain 15 candidates")
    source_sets = frame.groupby("method")["source_id"].apply(set)
    if len({frozenset(value) for value in source_sets}) != 1:
        raise ValueError("the five methods are not paired on the same 60 source IDs")
    return methods


def agreement(primary: pd.DataFrame, repeat: pd.DataFrame) -> dict:
    merged = primary.merge(
        repeat,
        on="canonical_id",
        suffixes=("_primary", "_repeat"),
        validate="one_to_one",
    )
    if len(merged) != 60:
        raise ValueError("repeat round must match exactly 60 primary candidates")
    result = {"repeated_pairs": 60, "type": "single-rater test-retest"}
    for field in SCORE_FIELDS:
        complete = merged[[f"{field}_primary", f"{field}_repeat"]].dropna()
        left = complete.iloc[:, 0].to_numpy(dtype=int)
        right = complete.iloc[:, 1].to_numpy(dtype=int)
        result[field] = {
            "n": len(complete),
            "quadratic_weighted_cohen_kappa": cohen_kappa(left, right, quadratic=True),
            "exact_agreement": float(np.mean(left == right)) if len(left) else None,
            "mean_absolute_difference": (
                float(np.mean(np.abs(left - right))) if len(left) else None
            ),
        }
    left_pass = merged["accepted_primary"].astype(int).to_numpy()
    right_pass = merged["accepted_repeat"].astype(int).to_numpy()
    result["pass_fail"] = {
        "n": len(merged),
        "unweighted_cohen_kappa": cohen_kappa(left_pass, right_pass, quadratic=False),
        "exact_agreement": float(np.mean(left_pass == right_pass)),
    }
    return result


def analyze(
    primary: pd.DataFrame, evaluations: pd.DataFrame, flux_method: str
) -> dict[str, object]:
    methods = validate_primary(primary, flux_method)
    baselines = [method for method in methods if method != flux_method]
    method_rows = []
    for method in methods:
        group = primary[primary["method"] == method]
        passed = int(group["accepted"].sum())
        low, high = wilson_interval(passed, len(group))
        method_rows.append(
            {
                "method": method,
                "passed": passed,
                "total": len(group),
                "pass_rate": passed / len(group),
                "wilson_95_low": low,
                "wilson_95_high": high,
                "unjudgeable_identity": int(
                    (group["identity_judgment_valid"] != "yes").sum()
                ),
                "missing_core": int(group["missing_core"].sum()),
            }
        )
    method_pass_rates = pd.DataFrame(method_rows)

    pass_comparisons = []
    for baseline in baselines:
        paired = primary[primary["method"].isin([flux_method, baseline])].pivot(
            index="source_id", columns="method", values="accepted"
        )
        flux = paired[flux_method].astype(int).to_numpy()
        other = paired[baseline].astype(int).to_numpy()
        flux_only, baseline_only, p_value = exact_mcnemar(flux, other)
        difference, low, high = bootstrap_difference(flux, other)
        pass_comparisons.append(
            {
                "baseline": baseline,
                "flux_only_pass": flux_only,
                "baseline_only_pass": baseline_only,
                "mcnemar_exact_two_sided_p": p_value,
                "paired_pass_rate_difference": difference,
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "bootstrap_seed": ANALYSIS_SEED,
            }
        )
    adjusted = holm_adjust(
        [row["mcnemar_exact_two_sided_p"] for row in pass_comparisons]
    )
    for row, value in zip(pass_comparisons, adjusted, strict=True):
        row["holm_adjusted_p"] = value
    paired_pass = pd.DataFrame(pass_comparisons)

    ordinal_rows = []
    for field in SCORE_FIELDS:
        dimension_rows = []
        for baseline in baselines:
            subset = primary[primary["method"].isin([flux_method, baseline])]
            if field == "identity_score":
                subset = subset[subset["identity_judgment_valid"] == "yes"]
            paired = subset.pivot(index="source_id", columns="method", values=field).dropna()
            flux = paired[flux_method].to_numpy(dtype=float)
            other = paired[baseline].to_numpy(dtype=float)
            n_nonzero, statistic, p_value = wilcoxon_exact(flux, other)
            difference, low, high = bootstrap_difference(flux, other)
            dimension_rows.append(
                {
                    "dimension": field,
                    "baseline": baseline,
                    "complete_pairs": len(paired),
                    "nonzero_pairs": n_nonzero,
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_exact_two_sided_p": p_value,
                    "paired_mean_difference": difference,
                    "bootstrap_95_low": low,
                    "bootstrap_95_high": high,
                    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                    "bootstrap_seed": ANALYSIS_SEED,
                }
            )
        corrected = holm_adjust(
            [row["wilcoxon_exact_two_sided_p"] for row in dimension_rows]
        )
        for row, value in zip(dimension_rows, corrected, strict=True):
            row["holm_adjusted_p_within_dimension"] = value
        ordinal_rows.extend(dimension_rows)
    ordinal = pd.DataFrame(ordinal_rows)

    score_long = primary.melt(
        id_vars=["method", "style_category"],
        value_vars=list(SCORE_FIELDS),
        var_name="dimension",
        value_name="score",
    )
    score_long_all = pd.concat(
        [score_long, score_long.assign(style_category="ALL")], ignore_index=True
    )
    score_distributions = (
        score_long_all.groupby(["method", "style_category", "dimension"])["score"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    style_pass_rates = (
        primary.groupby(["method", "style_category"])["accepted"]
        .agg(passed="sum", total="count", pass_rate="mean")
        .reset_index()
    )
    style_intervals = [
        wilson_interval(int(row.passed), int(row.total))
        for row in style_pass_rates.itertuples()
    ]
    style_pass_rates["wilson_95_low"] = [interval[0] for interval in style_intervals]
    style_pass_rates["wilson_95_high"] = [interval[1] for interval in style_intervals]

    expected_pairs = set(primary["canonical_id"])
    evaluations = evaluations.copy()
    evaluations["canonical_id"] = evaluations["method"] + ":" + evaluations["source_id"]
    if len(evaluations) != 300 or evaluations["canonical_id"].nunique() != 300:
        raise ValueError("automatic evaluations must contain exactly 300 unique pairs")
    if set(evaluations["canonical_id"]) != expected_pairs:
        raise ValueError("automatic evaluations do not exactly match primary human candidates")
    combined = primary.merge(
        evaluations,
        on=["canonical_id", "method", "source_id", "style_category"],
        validate="one_to_one",
    )
    alignment_rows = []
    auc_rows = []
    relations = {
        "dinov2_cosine": ("content_score", "style_removal_score"),
        "clip_cosine": ("content_score", "style_removal_score"),
        "arcface_cosine": ("identity_score", "style_removal_score"),
        "qwen_content_score": ("style_removal_score",),
        "qwen_style_removal_score": ("style_removal_score",),
        "qwen_identity_score": ("style_removal_score",),
    }
    for metric in AUTO_METRICS:
        for target in relations[metric]:
            for scope, group in [("pooled", combined), *combined.groupby("method")]:
                valid = group[[metric, target]].dropna()
                if metric == "arcface_cosine" and target == "identity_score":
                    valid = valid.loc[
                        group.loc[valid.index, "identity_judgment_valid"] == "yes"
                    ]
                alignment_rows.append(
                    {
                        "scope": scope,
                        "automatic_metric": metric,
                        "human_target": target,
                        "n": len(valid),
                        "spearman_rho": (
                            spearman(
                                valid[metric].to_numpy(dtype=float),
                                valid[target].to_numpy(dtype=float),
                            )
                            if len(valid)
                            else None
                        ),
                    }
                )
        for scope, group in [("pooled", combined), *combined.groupby("method")]:
            valid = group[[metric, "accepted"]].dropna()
            auc_rows.append(
                {
                    "scope": scope,
                    "automatic_metric": metric,
                    "n": len(valid),
                    "positive_passes": int(valid["accepted"].sum()),
                    "roc_auc_diagnostic": (
                        roc_auc(
                            valid[metric].to_numpy(dtype=float),
                            valid["accepted"].astype(int).to_numpy(),
                        )
                        if len(valid)
                        else None
                    ),
                }
            )
    metric_alignment = pd.DataFrame(alignment_rows)
    metric_auc = pd.DataFrame(auc_rows)
    arcface_missing = (
        combined.groupby("method")["arcface_cosine"]
        .apply(lambda values: int(values.isna().sum()))
        .to_dict()
    )
    arcface_no_face = (
        combined.assign(
            arcface_no_face=combined["arcface_status"].fillna("").str.startswith("no_face_")
        )
        .groupby("method")["arcface_no_face"]
        .sum()
        .astype(int)
        .to_dict()
    )
    arcface_status_counts = {
        method: {
            str(status): int(count)
            for status, count in group["arcface_status"]
            .fillna("missing_status")
            .value_counts()
            .sort_index()
            .items()
        }
        for method, group in combined.groupby("method")
    }
    primary_supported = all(
        row["paired_pass_rate_difference"] > 0
        and row["bootstrap_95_low"] > 0
        and row["holm_adjusted_p"] < 0.05
        for row in pass_comparisons
    )
    return {
        "method_pass_rates": method_pass_rates,
        "paired_pass": paired_pass,
        "ordinal": ordinal,
        "score_distributions": score_distributions,
        "style_pass_rates": style_pass_rates,
        "metric_alignment": metric_alignment,
        "metric_auc": metric_auc,
        "arcface_missing_by_method": arcface_missing,
        "arcface_no_face_by_method": arcface_no_face,
        "arcface_status_counts_by_method": arcface_status_counts,
        "primary_hypothesis_supported": primary_supported,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-scores", type=Path, required=True)
    parser.add_argument("--repeat-scores", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--flux-method", default="flux_kontext_native1024")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"refusing non-empty output directory: {args.output_dir}")
    try:
        key = load_key(args.private_key)
        primary = pd.DataFrame(load_scores(args.primary_scores, key, "primary"))
        repeat = pd.DataFrame(load_scores(args.repeat_scores, key, "repeat"))
        methods = validate_primary(primary, args.flux_method)
        repeat_counts = repeat.groupby(["method", "style_category"]).size()
        if len(repeat) != 60 or len(repeat_counts) != 20 or set(repeat_counts.tolist()) != {3}:
            raise ValueError("repeat round must contain three candidates in each method-style cell")
        evaluation_records = read_jsonl(args.evaluations, FormalEvaluationRecord)
        evaluations = pd.DataFrame(
            [record.model_dump(mode="json") for record in evaluation_records]
        )
        results = analyze(primary, evaluations, args.flux_method)
        agreement_result = agreement(primary, repeat)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        parser.error(str(exc))

    args.output_dir.mkdir(parents=True)
    primary.to_csv(args.output_dir / "unblinded-primary.csv", index=False)
    repeat.to_csv(args.output_dir / "unblinded-repeat.csv", index=False)
    table_files = {
        "method_pass_rates": "method-pass-rates.csv",
        "paired_pass": "paired-pass-comparisons.csv",
        "ordinal": "ordinal-score-comparisons.csv",
        "score_distributions": "score-distributions-by-method-style.csv",
        "style_pass_rates": "pass-rates-by-method-style.csv",
        "metric_alignment": "metric-alignment-spearman.csv",
        "metric_auc": "metric-pass-roc-auc-diagnostic.csv",
    }
    for key_name, filename in table_files.items():
        results[key_name].to_csv(args.output_dir / filename, index=False)
    (args.output_dir / "test-retest-agreement.json").write_text(
        json.dumps(agreement_result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    failure_counts = {
        "core_dimension_failures": {
            field: int(primary[field].sum())
            for field in (
                "content_dimension_failed",
                "style_dimension_failed",
                "identity_dimension_failed",
            )
        },
        "missing_core_candidates": int(primary["missing_core"].sum()),
        "identity_unjudgeable_or_missing": int(
            (primary["identity_judgment_valid"] != "yes").sum()
        ),
        "manual_failure_types": dict(
            sorted(
                Counter(
                    item
                    for value in primary["failure_types"]
                    for item in value.split(";")
                    if item
                ).items()
            )
        ),
        "manual_failure_type_reported_rows": int(primary["failure_types_reported"].sum()),
        "manual_failure_type_not_reported_rows": int(
            len(primary) - primary["failure_types_reported"].sum()
        ),
        "blank_failure_type_semantics": "not_reported",
        "arcface_missing_by_method": results["arcface_missing_by_method"],
        "arcface_no_face_by_method": results["arcface_no_face_by_method"],
        "arcface_status_counts_by_method": results["arcface_status_counts_by_method"],
        "arcface_missing_policy": "kept missing and counted; never imputed as zero",
    }
    (args.output_dir / "failure-counts.json").write_text(
        json.dumps(failure_counts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "schema": "face-destyle-formal-v1-heldout-analysis/v1",
        "methods": methods,
        "flux_method": args.flux_method,
        "primary_candidates": len(primary),
        "repeat_candidates": len(repeat),
        "acceptance_rule": (
            "content>=4 and style_removal>=4 and identity_judgment_valid=yes and identity>=4"
        ),
        "missing_core_policy": "candidate fails; no imputation",
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": ANALYSIS_SEED},
        "primary_hypothesis_supported": results["primary_hypothesis_supported"],
        "multiple_testing": {
            "pass": "Holm across four exact two-sided McNemar comparisons",
            "ordinal": "Holm across four Wilcoxon comparisons separately per dimension",
        },
        "metric_note": "Spearman and ROC-AUC are diagnostic; no thresholds or composite score.",
    }
    (args.output_dir / "analysis-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Completed frozen held-out analysis for {len(primary)} candidates in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
