# Origami V1 checkpoint 100 residual-Stage 2 review

## Run and result

The final inference-only diagnostic used the original fixed-holdout V1 checkpoint 100 outputs as
explicit inputs to Base FLUX's Origami Stage 2 prompt. It ran only the three V1 failures
`matv2-origami-002`, `011`, and `018`, at seed 42, 28 steps, and guidance 2.5. All 3/3 records and
images completed without a runner failure. This was a true chain:

`original -> frozen V1 checkpoint 100 -> Base FLUX residual-material Stage 2`

| Source | V1 checkpoint 100 | V1 -> Stage 2 | Strict result |
|---|---|---|---|
| `002` | Paper hair, beard, clothing, bust, support, and background geometry remain | Almost no useful material removal; the full construction remains folded paper | fail |
| `011` | Face improves, but scalp and garment remain paper | Face, scalp, and hair become photographic; the large folded garment remains dominant | partial/fail |
| `018` | Face improves, but paper hair and clothing remain | Only minor facial refinement; paper hair, eyebrows, clothing, and scene geometry remain | fail |

The diagnostic rescues 0/3 under the existing strict full-frame rule. It therefore does not raise
V1 checkpoint 100 above the previously reviewed approximate 3/6 holdout result. `011` confirms that
a residual edit can help a local region, but also confirms why face-only improvement is not a
full-frame pass.

## Routing consequence

Freeze V1 checkpoint 100 as an optional limited adapter. A residual Stage 2 may be requested after
human review, but it is not a guaranteed rescue route. If paper remains across hair, clothing,
bust, support, or background, route to a manually authorized closed teacher or record an explicit
failure. Do not resume V2/V2.1, train a new adapter, or describe the residual edit as improving the
aggregate strict pass count.

## External evidence location

No images, logs, or weights are copied into Git. The returned local folder is
`origami-v1ckpt100-residual-stage2-hard3-seed42`; the AutoDL output root was
`/root/autodl-tmp/face-destyle/outputs/origami-v1ckpt100-residual-stage2-hard3-seed42`.
