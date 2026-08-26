# Multistyle processing-routing completion manifests

`missing_stage12_sources.jsonl` freezes the compact 24-source set used to fill the actual
Stage 1/true-sequential-Stage 2 coverage gap:

| Style | Sources | Selection |
|---|---:|---|
| Comic | 6 | five existing pilot sources plus `synthetic-comic-006` |
| Ink | 6 | five existing pilot sources plus `met-335049` |
| Watercolor | 6 | five existing pilot sources plus `synthetic-watercolor-005` |
| Needle-felt | 6 | diverse replacement sources `001`, `002`, `006`, `007`, `008`, `011` |

The first three groups intentionally reuse existing authorized non-test inputs so the new result
isolates the missing sequential edit rather than starting another acquisition round. Needle-felt
uses the clearer replacement bank from `extensions/material_styles_v2`, not the earlier ambiguous
material-v1 candidates.

All paths resolve against the private `Face-DeStyle-Data` root. The records use `split=extension`
because this is a new lightweight routing-completion experiment, not a reopening of formal-v1.
Images remain outside Git.

Run both stages for all 24 records. Stage 2 must use the Stage 1 `records.jsonl` through
`--input-records`; running a Stage 2 prompt against the original manifest does not satisfy this
experiment.

`non_origami_validation_137.jsonl` is the subsequent larger routing-validation bank. Origami is
excluded because its V1/V2/V2.1 and residual-edit branch is already frozen. The bank contains:

| Style | Sources | Selection |
|---|---:|---|
| Comic | 24 | all formal-v1 sources not used in the six-source routing-gap run |
| Ink | 24 | all formal-v1 sources not used in the six-source routing-gap run |
| Watercolor | 24 | all formal-v1 sources not used in the six-source routing-gap run |
| 3D cartoon | 24 | accepted pair-bank sources `001`--`024`, including declared holdouts |
| Clay | 24 | all accepted material-v2 candidate and holdout sources |
| Needle-felt | 17 | all 12 accepted material-v2 sources plus the five accepted material-v1 pilot sources |

The smaller Needle-felt count is intentional: rejected material-v1 sources are not reintroduced
merely to make the table balanced. The 137-source run generates both stages for every source, or
274 output images total, so review can measure Stage 2 rescue and regression without selecting the
second-stage subset after seeing results.

The manifest is generated deterministically by
`scripts/build_multistyle_routing_validation_manifest.py`. Rebuilding is unnecessary during normal
execution; it exists to make selection provenance clear.
