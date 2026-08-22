# Formal-v1 reduced post-unblinding replication analysis

## Status and scope

This is a **reduced post-unblinding replication analysis** (精简复刻评估), not the preregistered
formal-v1 300-candidate confirmatory held-out test. The operator selected 32 test sources after
unblinding, eight per style, while retaining all five methods for every source. Selection prioritized
sources with the greatest number of already-complete human ratings and used seed `20260822` to break
ties. Of the 160 ratings, 99 were inherited and 61 were completed after source selection. The
operator canceled the repeat round, so no test-retest reliability estimate exists.

The original 300-candidate review material was not overwritten. No test image was visually opened
during machine validation, freezing, unblinding verification, or statistical analysis.

## Validation and frozen decision rule

Machine validation confirmed exactly 160 unique blind IDs, 32 unique sources, all five methods for
each source, 32 candidates per method, and eight sources/40 candidates per style. All three core
scores were complete integers from 0 to 5. Identity judgment was `yes` for 150 candidates and `no`
for 10. The 160-row selection mapping matched all method/source/style fields in the original sealed
private key with zero discrepancies.

A candidate passed only if content and style-removal scores were both at least 4, identity was
judgeable, and identity score was at least 4. Unjudgeable identity failed. No values were imputed,
no composite score was constructed, and no threshold was selected from this reduced test subset.
Blank failure types are reported as `not_reported`.

## Method pass rates

| Method | Passed / 32 | Pass rate | Wilson 95% CI |
|---|---:|---:|---:|
| FLUX Kontext native 1024 | 22 | 68.75% | 51.43%–82.05% |
| Global Canny 0.4 | 0 | 0.00% | 0.00%–10.72% |
| Prompt adaptive | 0 | 0.00% | 0.00%–10.72% |
| Prompt generic | 1 | 3.12% | 0.55%–15.74% |
| Region Canny | 1 | 3.12% | 0.55%–15.74% |

Within this completion-informed reduced sample, FLUX passed 22 sources. Its style-specific counts
were 4/8 for 3D cartoon and 6/8 each for comic, ink, and watercolor. Generic and Region Canny each
passed one 3D-cartoon source; the other SDXL method-style cells had zero passes.

## Paired pass comparisons

All comparisons contain 32 complete source pairs. Exact McNemar tests are two-sided; the four
comparisons are Holm-adjusted. Bootstrap intervals use 20,000 source-level resamples with seed
`20260821`.

| Baseline | FLUX-only / baseline-only passes | Paired difference | Bootstrap 95% CI | Exact p | Holm p |
|---|---:|---:|---:|---:|---:|
| Global Canny 0.4 | 22 / 0 | 68.75 pp | 53.12–84.38 pp | 4.77e-7 | 1.91e-6 |
| Prompt adaptive | 22 / 0 | 68.75 pp | 53.12–84.38 pp | 4.77e-7 | 1.91e-6 |
| Prompt generic | 21 / 0 | 65.62 pp | 50.00–81.25 pp | 9.54e-7 | 1.91e-6 |
| Region Canny | 21 / 0 | 65.62 pp | 50.00–81.25 pp | 9.54e-7 | 1.91e-6 |

Paired exact Wilcoxon comparisons likewise favored FLUX over every baseline for content,
style-removal, and identity scores after Holm adjustment within each dimension (all adjusted
`p <= 1.58e-6`). Identity comparisons used only sources where both candidates had a valid identity
judgment, yielding 27–32 complete pairs depending on the baseline. These are reduced-sample results,
not confirmation of the original preregistered primary hypothesis.

## Failure reporting and automatic metrics

Core rubric dimensions failed for 117 content ratings, 131 style-removal ratings, and 125 identity
ratings. Manual failure labels are multi-select: artistic contour residual 115, material render
residual 61, identity drift 43, background drift 36, structure drift 22, and no usable face 1.
Failure type was blank, and therefore explicitly `not_reported`, for 26 candidates.

No complete existing test automatic-metric mapping was found or supplied for the selected 160
candidates. Consequently the preplanned Spearman and ROC-AUC diagnostics were not run. No new model
or metric inference was started. If a complete existing evaluation JSONL becomes available,
`scripts/analyze_reduced_heldout.py --evaluations ...` can crop it to the frozen 160 mappings;
ArcFace no-face remains missing and is counted separately rather than filled with zero.

## Interpretation limitation

The large paired differences in this subset are descriptive evidence that FLUX performed better
than the four SDXL baselines among the selected sources. They do not remove the central design
limitation: source selection occurred after unblinding and explicitly used rating-completion
information. Without the unselected sources and without repeat scoring, sampling uncertainty and
reliability cannot be interpreted as though this were the planned full confirmatory test. This
analysis is an exploratory reduced replication only.

Reproducible machine-readable tables, the freeze record, and analysis metadata are in
`docs/results/formal_v1_reduced_32/`.
