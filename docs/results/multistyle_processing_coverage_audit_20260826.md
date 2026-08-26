# Multistyle processing coverage audit (2026-08-26)

> Historical pre-run audit. Every gap identified below was resolved later on 2026-08-26. See
> [`multistyle_project_closure_20260826.md`](multistyle_project_closure_20260826.md) for the final
> results and routes. Do not treat the “remaining gap” column as active work.

## Correction to the earlier closure

At audit time, the earlier closure had completed the style-specific LoRA evidence narrative but not
the executable route or matched Stage 1/true-sequential-Stage 2 experiment. That processing gap has
since been completed without another LoRA campaign.

## Actual completed coverage

| Style | Base / Stage 1 evidence | Stage 2 evidence | Targeted adaptation | Remaining gap |
|---|---|---|---|---|
| Comic | Base FLUX ran on pilot, calibration, and test sources | no true sequential Stage 2 | none needed | run a compact matched Stage 1/Stage 2 set and route it |
| Ink | Base FLUX ran on pilot, calibration, and test sources | no true sequential Stage 2 | none needed | run a compact matched Stage 1/Stage 2 set and route it |
| Watercolor | Base FLUX ran on pilot, calibration, and test sources | no true sequential Stage 2 | none needed | run a compact matched Stage 1/Stage 2 set and route it |
| Needle-felt | five-source material-v1 Base run | the old Stage 2 edited the original source again; it was not sequential | none planned | run replacement sources through a true Stage 1-to-Stage 2 chain |
| Clay | 19 material-v2 candidate Stage 1 outputs | 12/12 reviewed true sequential outputs | one strict teacher pair; no LoRA | no generation gap for this stage; integrate existing decisions into the router |
| 3D cartoon | five-source Base pilot and 27 pair-bank Stage 1 outputs | 8/8 reviewed true sequential outputs, with no strict target | negative eight-pair LoRA smoke and one teacher pair | no repeat run; integrate the negative route |
| Origami | 24 pair-bank Stage 1 outputs | 10/10 reviewed true sequential outputs | V1-100 about 3/6; V2/V2.1 negative | test the still-unrun V1-output-to-residual-Stage-2 fallback, then integrate it |

The formal-v1 fallback result is also not an executable router: existing SDXL candidates rescued
0/3 FLUX calibration failures. It supports rejecting that particular fallback, not claiming that
multistyle routing was implemented.

## Frozen missing-coverage set

The repository-owned manifest is
`data/manifests/multistyle-routing/missing_stage12_sources.jsonl`. It contains exactly 24 accepted,
checksum-pinned records: six each for Comic, Ink, Watercolor, and replacement Needle-felt.

The experiment runs both stages on every record under the same original-BF16 FLUX Kontext settings:
native 1024 square, seed 42, 28 steps, guidance 2.5, model offload, and the declared style prompts.
Running Stage 2 on all records measures both rescue and harm; selecting Stage 2 only after seeing
Stage 1 would not reveal how often the second edit introduces drift.

## Review outputs used after generation

For each source, compare `source | Stage 1 | Stage 1 -> Stage 2` and record:

- `stage1_pass` and `stage2_pass`;
- full-frame style/material residual;
- subject, face, pose, expression, clothing, crop, and background preservation;
- `stage2_rescue` when Stage 1 fails and Stage 2 passes;
- `stage2_regression` when Stage 1 passes and Stage 2 fails;
- one final route: `accept_stage1`, `accept_stage2`, or `explicit_failure`.

No blind round, formal metric suite, LoRA training, or new source generation was used for this
compact completion experiment.
