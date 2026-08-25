# Origami LoRA V2 six-holdout review (2026-08-25)

## Outcome

The 51-pair V2 intervention did not improve the strict held-out result. The fixed comparison
completed 6/6 RGB 1024-square outputs for Base, frozen V1 checkpoint 100, and V2 checkpoints
50/100/150/200. All methods used seed 42, 28 inference steps, guidance 2.5, and LoRA scale 1.0.

Under the same full-frame standard used for V1, the conservative strict pass counts are:

| Method | Strict passes | Interpretation |
|---|---:|---|
| Base | 1/6 | `023` only |
| V1 checkpoint 100 | 3/6 | `007`, `023`, `030`; remains the best adapter |
| V2 checkpoint 50 | 0/6 | too conservative; conspicuous paper remains throughout |
| V2 checkpoint 100 | 2/6 | `007`, `023` |
| V2 checkpoint 150 | 2/6 | `007`, `023` |
| V2 checkpoint 200 | 2/6 strict | `007`, `023`; `030` removes paper but drifts in age, beard, and expression |

The useful target of at least 4/6 strict passes was not met. None of the required hard failures
`002`, `011`, or `018` was rescued. The expansion therefore gives a negative/mixed result rather
than an improved adapter.

## Per-source review

| Holdout | Base | V1-100 | V2-50 | V2-100 | V2-150 | V2-200 | Full-frame rationale |
|---|---|---|---|---|---|---|---|
| `002` | fail | fail | fail | fail | fail | fail | Folded hair, beard, layered clothing, bust, and pedestal remain at every checkpoint. Later V2 mainly smooths facial skin. |
| `007` | fail | pass | fail | pass | pass | pass | V2 begins removing face and hair material by 100; later checkpoints stay natural enough while retaining pose and broad facial evidence. |
| `011` | fail | fail | fail | fail | fail | fail | Scalp, head planes, neck, and garment remain constructed from paper even when eyes and central face become smoother. |
| `018` | fail | fail | fail | fail | fail | fail | V2-100 onward naturalizes the central face, but large folded hair/headwear and the complete shoulder garment remain paper. |
| `023` | near/pass | pass | fail | pass | pass | pass | V2-100 onward produces natural skin, hair, hood, and garment while retaining gaze, earring, palette, and composition. |
| `030` | fail | pass | fail | fail | fail | fail strict | V2-200 finally removes scalp, beard, and garment paper, but changes gray facial hair to black, reduces apparent age, and introduces a broader smile. |

## Interpretation

V2 shows a clear progression from conservative reconstruction at checkpoint 50 toward stronger
material removal by checkpoint 200, so the adapter loaded and learned from the new data. That
progression does not translate into the intended generalization boundary: the large hair/clothing
case `018`, severe scalp/garment case `011`, and full bust/pedestal case `002` all retain their
original construction. The only additional style-removal signal at V2-200 is `030`, and it comes
with enough identity, age, beard-color, and expression drift to fail the strict rule.

Do not select a V2 checkpoint as an improvement over V1 checkpoint 100, do not report 3/6 for
V2-200 by ignoring the `030` drift, and do not automatically continue V2 past 200 steps. The honest
result is that adding 28 carefully reviewed difficult pairs and region-specific captions was not
sufficient to improve this six-source generalization test.

## Prompt-alignment diagnostic

A subsequent source-specific prompt diagnostic kept V2 checkpoint 200, seed 42, 28 inference
steps, guidance 2.5, and LoRA scale 1.0 frozen. It changed only the instruction for the unresolved
holdouts `002`, `011`, and `018`, using full-subject region language aligned with the V2 training
captions.

| Holdout | Matched-prompt result | Rationale |
|---|---|---|
| `002` | fail, materially improved | Face, hair, and gray beard became natural, but the garment, lower bust, pedestal, support, and a rear hair ornament retained folded-paper geometry. |
| `011` | pass | Bald scalp, face, neck, and complete garment became plausible skin and fabric while preserving the pose, gaze, palette, and composition. |
| `018` | fail, partially improved | The face became natural, but the large outer hair/headwear mass and complete shoulder garment remained folded paper. |

The diagnostic rescues one of the three hard cases and demonstrates that train/evaluation
instruction mismatch contributed to the original result. It does not establish that prompt
alignment alone solves V2: only `011` is a new strict pass, while `002` and `018` still expose a
full-subject material-removal limitation. Combined with the unchanged `007` and `023` passes, this
is approximately 3/6 rather than the required 4/6. V1 checkpoint 100 therefore remains the frozen
selected adapter.

## Artifacts

- downloaded archive:
  `/Users/pot/Documents/大创/实验归档/returned-runs/origami-lora-heldout-v2-base-v1ckpt100-v2ckpt50-100-150-200-seed42.zip`
- local seven-column review sheets:
  `/Users/pot/Documents/大创/实验归档/origami-lora-heldout-v2-review-20260825`
- prompt-alignment archive:
  `/Users/pot/Documents/大创/实验归档/returned-runs/origami-lora-v2-prompt-alignment-hard3-seed42.zip`
- prompt-alignment review sheets:
  `/Users/pot/Documents/大创/实验归档/origami-lora-v2-prompt-alignment-review-20260825`
- compact presentation set:
  `/Users/pot/Documents/大创/实验归档/showcase-20260825`
- V1 selected weight SHA-256:
  `06ab9433e341713aaaa0edb11849db5e687b47ad0ca930c121cea49277eca7c4`
- V2 checkpoint-200 weight SHA-256:
  `a73440e61c3f143560b4c024c639e1c9ad71b7ba004f4e05601a486c345817f5`
