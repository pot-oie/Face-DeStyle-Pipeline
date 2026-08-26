import csv
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REVIEW = (
    ROOT / "docs/results/multistyle_routing_validation_137_review_20260827.csv"
)


def test_final_multistyle_routing_policy_covers_supported_styles():
    policy = yaml.safe_load(
        (ROOT / "configs/multistyle_routing.yaml").read_text(encoding="utf-8")
    )

    assert policy["review_required"] is True
    assert policy["fallback_when_unacceptable"] == "explicit_failure"
    assert set(policy["styles"]) == {
        "comic",
        "ink",
        "watercolor",
        "3d_cartoon",
        "clay",
        "needle_felt",
        "origami",
    }
    assert policy["styles"]["comic"]["default_route"] == "flux_stage1"
    assert policy["styles"]["clay"]["default_route"] == (
        "flux_stage1_then_flux_stage2"
    )
    assert policy["styles"]["origami"]["optional_next_route"] == (
        "frozen_v1_checkpoint_100_limited"
    )


def test_final_routing_review_has_one_terminal_decision_per_source():
    with REVIEW.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 137
    assert len({row["source_id"] for row in rows}) == 137
    assert Counter(row["selected_route"] for row in rows) == {
        "accept_stage1": 66,
        "accept_stage2": 5,
        "explicit_failure": 66,
    }
    assert sum(row["stage2_rescue"] == "yes" for row in rows) == 5
    assert sum(row["stage2_regression"] == "yes" for row in rows) == 0
