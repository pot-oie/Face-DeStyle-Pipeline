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
