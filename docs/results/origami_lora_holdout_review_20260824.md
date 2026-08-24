# Origami LoRA six-holdout review

The first strict Origami adapter trained from 23 accepted closed-teacher pairs. Training completed
at rank 16 with checkpoints 100, 200, and 300. The held-out package is
`origami-lora-heldout-base-ckpt100-200-300-seed42.zip` with SHA-256
`099b89ff299f761567fa5ea79c0fe0b30dbc701974ba5b58dfaf1441695f00fa`.

All four methods produced 6/6 RGB 1024-square images with zero runner failures. Controlled settings
were seed 42, 28 inference steps, guidance 2.5, and LoRA scale 1.0. The protected holdouts were
`matv2-origami-002`, `007`, `011`, `018`, `023`, and `030`.

| Holdout | Base | Checkpoint 100 | Strict review |
|---|---|---|---|
| `002` | fail | fail | paper hair, clothing, and bust remain |
| `007` | fail | pass | skin, hair, and clothing naturalize |
| `011` | fail | partial/fail | face improves; scalp and garment remain paper-like |
| `018` | fail | partial/fail | face improves; paper hair and clothing remain |
| `023` | near/pass | pass | stable identity, gaze, hair, hood, and garment |
| `030` | fail | pass | skin, beard, scalp, and clothing naturalize |

The conservative aggregate is roughly 1/6 strict pass for Base and 3/6 for each LoRA checkpoint.
Checkpoint 200 and 300 did not rescue another holdout. They progressively altered face fullness,
age, mouth shape, or expression on already-successful cases. Checkpoint 100 is therefore selected
as the least-overfit useful adapter:

```text
/root/autodl-tmp/face-destyle/outputs/origami-destyle-lora-teacher23-r16-steps300/checkpoint-100/pytorch_lora_weights.safetensors
SHA-256 06ab9433e341713aaaa0edb11849db5e687b47ad0ca930c121cea49277eca7c4
```

This is a positive but limited result: paired adaptation improves strong material removal on three
genuine unseen sources, but the current data does not cover severe folded hair/headwear, scalp,
clothing, bust, and pedestal cases well enough. Continuing the same run is rejected. The next test
is a fresh adapter built from the original 23 pairs plus approximately 24--30 new strictly reviewed
hard pairs, while keeping these six holdouts unchanged.
