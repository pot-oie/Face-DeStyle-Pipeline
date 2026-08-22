#!/usr/bin/env python3
"""Analyze the frozen 32-source reduced post-unblinding held-out review."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from analyze_heldout_test import (
    ANALYSIS_SEED,
    AUTO_METRICS,
    BOOTSTRAP_RESAMPLES,
    SCORE_FIELDS,
    bootstrap_difference,
    exact_mcnemar,
    holm_adjust,
    roc_auc,
    spearman,
    wilcoxon_exact,
    wilson_interval,
)

from face_destyle.data.metadata import read_jsonl
from face_destyle.reduced_heldout import (
    METHODS,
    validate_private_key_mapping,
    validate_reduced_review,
)
from face_destyle.schemas import FormalEvaluationRecord

FLUX_METHOD = "flux_kontext_native1024"


def verify_freeze(validated: dict[str, object], freeze_path: Path) -> dict[str, object]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema") != "face-destyle-reduced-post-unblinding-freeze/v1":
        raise ValueError("freeze record has the wrong schema")
    if freeze.get("input_sha256") != validated["input_sha256"]:
        raise ValueError("reduced review inputs no longer match the freeze record")
    return freeze


def analyze_human(primary: pd.DataFrame, flux_method: str) -> dict[str, object]:
    methods = sorted(primary["method"].unique())
    if len(primary) != 160 or methods != sorted(METHODS) or flux_method not in methods:
        raise ValueError("reduced analysis requires the frozen 160-row five-method matrix")
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
                "identity_unjudgeable": int((group["identity_judgment_valid"] != "yes").sum()),
                "missing_core": int(group["missing_core"].sum()),
            }
        )

    pass_rows = []
    for baseline in baselines:
        paired = primary[primary["method"].isin([flux_method, baseline])].pivot(
            index="source_id", columns="method", values="accepted"
        )
        flux = paired[flux_method].astype(int).to_numpy()
        other = paired[baseline].astype(int).to_numpy()
        flux_only, baseline_only, p_value = exact_mcnemar(flux, other)
        difference, low, high = bootstrap_difference(flux, other)
        pass_rows.append(
            {
                "baseline": baseline,
                "complete_source_pairs": len(paired),
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
    for row, adjusted in zip(
        pass_rows,
        holm_adjust([row["mcnemar_exact_two_sided_p"] for row in pass_rows]),
        strict=True,
    ):
        row["holm_adjusted_p"] = adjusted

    ordinal_rows = []
    for dimension in SCORE_FIELDS:
        dimension_rows = []
        for baseline in baselines:
            subset = primary[primary["method"].isin([flux_method, baseline])]
            if dimension == "identity_score":
                subset = subset[subset["identity_judgment_valid"] == "yes"]
            paired = subset.pivot(index="source_id", columns="method", values=dimension).dropna()
            flux = paired[flux_method].to_numpy(dtype=float)
            other = paired[baseline].to_numpy(dtype=float)
            nonzero, statistic, p_value = wilcoxon_exact(flux, other)
            difference, low, high = bootstrap_difference(flux, other)
            dimension_rows.append(
                {
                    "dimension": dimension,
                    "baseline": baseline,
                    "complete_pairs": len(paired),
                    "nonzero_pairs": nonzero,
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_exact_two_sided_p": p_value,
                    "paired_mean_difference": difference,
                    "bootstrap_95_low": low,
                    "bootstrap_95_high": high,
                    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                    "bootstrap_seed": ANALYSIS_SEED,
                }
            )
        for row, adjusted in zip(
            dimension_rows,
            holm_adjust([row["wilcoxon_exact_two_sided_p"] for row in dimension_rows]),
            strict=True,
        ):
            row["holm_adjusted_p_within_dimension"] = adjusted
        ordinal_rows.extend(dimension_rows)

    score_long = primary.melt(
        id_vars=["method", "style_category"],
        value_vars=list(SCORE_FIELDS),
        var_name="dimension",
        value_name="score",
    )
    score_distributions = pd.concat(
        [score_long, score_long.assign(style_category="ALL")], ignore_index=True
    )
    score_distributions = (
        score_distributions.groupby(["method", "style_category", "dimension"])["score"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    style_rates = (
        primary.groupby(["method", "style_category"])["accepted"]
        .agg(passed="sum", total="count", pass_rate="mean")
        .reset_index()
    )
    intervals = [
        wilson_interval(int(row.passed), int(row.total)) for row in style_rates.itertuples()
    ]
    style_rates["wilson_95_low"] = [value[0] for value in intervals]
    style_rates["wilson_95_high"] = [value[1] for value in intervals]

    failure_counter = Counter(
        item or "not_reported"
        for value in primary["failure_types"]
        for item in (value.split(";") if value else [""])
    )
    failures = pd.DataFrame(
        [
            {"failure_type": name, "candidate_count": count}
            for name, count in sorted(failure_counter.items())
        ]
    )
    return {
        "method_pass_rates": pd.DataFrame(method_rows),
        "paired_pass": pd.DataFrame(pass_rows),
        "ordinal": pd.DataFrame(ordinal_rows),
        "score_distributions": score_distributions,
        "style_pass_rates": style_rates,
        "failure_types": failures,
    }


def analyze_metrics(primary: pd.DataFrame, path: Path) -> dict[str, object]:
    records = read_jsonl(path, FormalEvaluationRecord)
    evaluations = pd.DataFrame([record.model_dump(mode="json") for record in records])
    evaluations["canonical_id"] = evaluations["method"] + ":" + evaluations["source_id"]
    if evaluations["canonical_id"].duplicated().any():
        raise ValueError("automatic evaluations contain duplicate method-source pairs")
    selected_ids = set(primary["canonical_id"])
    cropped = evaluations[evaluations["canonical_id"].isin(selected_ids)].copy()
    if len(cropped) != 160 or set(cropped["canonical_id"]) != selected_ids:
        raise ValueError("automatic evaluations do not completely map to the selected 160 rows")
    combined = primary.merge(
        cropped,
        on=["canonical_id", "method", "source_id", "style_category"],
        validate="one_to_one",
    )
    relations = {
        "dinov2_cosine": ("content_score", "style_removal_score"),
        "clip_cosine": ("content_score", "style_removal_score"),
        "arcface_cosine": ("identity_score", "style_removal_score"),
        "qwen_content_score": ("style_removal_score",),
        "qwen_style_removal_score": ("style_removal_score",),
        "qwen_identity_score": ("style_removal_score",),
    }
    alignment_rows = []
    auc_rows = []
    scopes = [("pooled", combined), *combined.groupby("method")]
    for metric in AUTO_METRICS:
        for target in relations[metric]:
            for scope, group in scopes:
                valid = group[[metric, target]].dropna()
                if metric == "arcface_cosine" and target == "identity_score":
                    valid = valid[group.loc[valid.index, "identity_judgment_valid"] == "yes"]
                alignment_rows.append(
                    {
                        "scope": scope,
                        "automatic_metric": metric,
                        "human_target": target,
                        "n": len(valid),
                        "spearman_rho": (
                            spearman(valid[metric].to_numpy(), valid[target].to_numpy())
                            if len(valid)
                            else None
                        ),
                    }
                )
        for scope, group in scopes:
            valid = group[[metric, "accepted"]].dropna()
            auc_rows.append(
                {
                    "scope": scope,
                    "automatic_metric": metric,
                    "n": len(valid),
                    "positive_passes": int(valid["accepted"].sum()),
                    "roc_auc_diagnostic": (
                        roc_auc(valid[metric].to_numpy(), valid["accepted"].astype(int).to_numpy())
                        if len(valid)
                        else None
                    ),
                }
            )
    arcface = {
        "missing_by_method": combined.groupby("method")["arcface_cosine"]
        .apply(lambda values: int(values.isna().sum()))
        .to_dict(),
        "no_face_by_method": combined.assign(
            no_face=combined["arcface_status"].fillna("").str.startswith("no_face_")
        )
        .groupby("method")["no_face"]
        .sum()
        .astype(int)
        .to_dict(),
        "missing_policy": "kept missing and counted; never imputed as zero",
    }
    return {
        "metric_alignment": pd.DataFrame(alignment_rows),
        "metric_auc": pd.DataFrame(auc_rows),
        "arcface": arcface,
        "evaluation_records_total": len(evaluations),
        "evaluation_records_selected": len(cropped),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--freeze-record", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluations", type=Path)
    parser.add_argument("--flux-method", default=FLUX_METHOD)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"refusing non-empty output directory: {args.output_dir}")
    try:
        validated = validate_reduced_review(args.review_dir)
        verify_freeze(validated, args.freeze_record)
        key_verification = validate_private_key_mapping(args.private_key, validated["selection"])
        primary = pd.DataFrame(validated["rows"])
        results = analyze_human(primary, args.flux_method)
        metrics = analyze_metrics(primary, args.evaluations) if args.evaluations else None
    except (FileNotFoundError, KeyError, ValueError) as exc:
        parser.error(str(exc))

    args.output_dir.mkdir(parents=True)
    primary.drop(columns=["failure_types_reported"]).to_csv(
        args.output_dir / "unblinded-human-scores.csv", index=False
    )
    files = {
        "method_pass_rates": "method-pass-rates.csv",
        "paired_pass": "paired-pass-comparisons.csv",
        "ordinal": "ordinal-score-comparisons.csv",
        "score_distributions": "score-distributions-by-method-style.csv",
        "style_pass_rates": "pass-rates-by-method-style.csv",
        "failure_types": "failure-types.csv",
    }
    for key, filename in files.items():
        results[key].to_csv(args.output_dir / filename, index=False)
    metric_status: dict[str, object]
    if metrics:
        metrics["metric_alignment"].to_csv(
            args.output_dir / "metric-alignment-spearman.csv", index=False
        )
        metrics["metric_auc"].to_csv(
            args.output_dir / "metric-pass-roc-auc-diagnostic.csv", index=False
        )
        metric_status = {
            "status": "complete_existing_metrics_cropped_to_selected_sources",
            "source": str(args.evaluations.resolve()),
            "evaluation_records_total": metrics["evaluation_records_total"],
            "evaluation_records_selected": metrics["evaluation_records_selected"],
            "arcface": metrics["arcface"],
            "threshold_selection": "not performed",
            "composite_score": "not constructed",
        }
    else:
        metric_status = {
            "status": "not_run_no_complete_existing_test_metric_mapping_supplied",
            "new_metric_computation": "not performed",
            "threshold_selection": "not performed",
            "composite_score": "not constructed",
        }
    (args.output_dir / "automatic-metric-status.json").write_text(
        json.dumps(metric_status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "schema": "face-destyle-reduced-post-unblinding-analysis/v1",
        "analysis_label": "reduced post-unblinding replication analysis",
        "candidate_count": 160,
        "source_count": 32,
        "methods": sorted(primary["method"].unique()),
        "repeat_reliability": "not_available_operator_cancelled_repeat",
        "selection_disclosure": (
            "Sources were selected after unblinding and selection prioritized existing human "
            "score completion, with seed 20260822 for ties."
        ),
        "interpretation": (
            "Exploratory reduced replication; not equivalent to the preregistered "
            "300-candidate confirmatory held-out test."
        ),
        "acceptance_rule": (
            "content>=4 and style_removal>=4 and identity_judgment_valid=yes and identity>=4"
        ),
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": ANALYSIS_SEED},
        "multiple_testing": {
            "pass": "Holm across four exact two-sided McNemar comparisons",
            "ordinal": "Holm across four Wilcoxon comparisons separately per dimension",
        },
        "automatic_metrics": metric_status,
        "private_key_unblinding_verification": key_verification,
    }
    (args.output_dir / "analysis-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Completed reduced analysis for 160 candidates in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
