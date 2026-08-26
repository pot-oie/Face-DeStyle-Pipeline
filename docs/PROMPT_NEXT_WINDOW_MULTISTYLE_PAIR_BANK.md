# Prompt for the next main implementation window

> Superseded on 2026-08-26 by
> `docs/PROMPT_NEXT_WINDOW_MULTISTYLE_ROUTING_AND_CLOSURE.md`. The V2 and V2.1 runs completed without
> beating Origami V1; do not execute the training plan below as an active task.

Continue `Face-DeStyle-Pipeline` from the completed first Origami LoRA experiment and the completed,
independently reviewed hard-pair expansion. The operator has temporarily shut down AutoDL. Do not
attempt remote commands until the operator explicitly reports that the instance is running again.

Local repository: `/Users/pot/Github/Face-DeStyle-Pipeline`

Local dataset: `/Users/pot/Documents/大创/Face-DeStyle-Data`

AutoDL project root when online: `/root/autodl-tmp/face-destyle`

First read completely:

1. `AGENTS.md`
2. `docs/HANDOFF_MULTISTYLE_PAIR_BANK_AND_LORA.md`
3. `docs/PROMPT_ORIGAMI_HARD_PAIR_EXPANSION_V2.md`
4. `docs/results/multistyle_pair_bank_stage2_review_20260824.md`
5. `docs/results/origami_lora_holdout_review_20260824.md`
6. `docs/results/origami_hard_pairs_v2_review_20260825.md`
7. `scripts/build_pair_bank_lora_dataset.py`
8. `scripts/run_origami_lora_holdout_eval.sh`
9. `data/manifests/multistyle-pair-bank/origami_target_selection_v1.csv`

## Completed state

- The original strict Origami dataset contains 23 accepted source/closed-teacher pairs.
- The withdrawn `origami-lora-pairs-v1-20` contains residual paper and must never be trained.
- The 23-pair rank-16 Kontext LoRA completed 300 steps and saved checkpoints 100/200/300.
- A fixed six-holdout comparison completed Base plus all three checkpoints with 6/6 images and zero
  failures per method.
- Strict full-frame review found approximately 1/6 pass for Base and 3/6 for each LoRA checkpoint.
- Checkpoint 100 is frozen as the selected first adapter. Later checkpoints did not increase the
  pass count and caused more identity/expression drift.
- Selected weight:
  `/root/autodl-tmp/face-destyle/outputs/origami-destyle-lora-teacher23-r16-steps300/checkpoint-100/pytorch_lora_weights.safetensors`
- Selected SHA-256:
  `06ab9433e341713aaaa0edb11849db5e687b47ad0ca930c121cea49277eca7c4`
- Holdouts `007`, `023`, and `030` passed. `011` and `018` remained partial failures; `002` remained
  a strong failure. All six stay evaluation-only.
- More training steps are rejected as the next intervention. The active bottleneck is targeted
  difficult-pair coverage and stronger region-specific instructions.
- The expansion at
  `/Users/pot/Documents/大创/Face-DeStyle-Data/extensions/origami_hard_pairs_v2` contains 36 new
  sources and 30 generator-accepted teachers. Independent review accepts 28 for training.
- Exclude `origami-hard-v2-021` and `origami-hard-v2-023`: both remove a visible bust/pedestal and
  therefore change composition. Do not take all 30 rows from the delivered CSV.
- The accepted 28 plus the original 23 produce an exact 51-pair V2 dataset. This is sufficient; do
  not generate another image batch before testing it.
- The delivered CSV uses slightly different columns and absolute paths. Treat it as provenance
  input, not the final training manifest.

## First concrete task

1. Check the working tree and preserve unrelated user changes.
2. Create a tracked, portable V2 selection manifest containing the exact 28 accepted IDs from
   `docs/results/origami_hard_pairs_v2_review_20260825.md`, with source-relative paths, decisions,
   difficulty tags, and reviewer notes. Do not commit bulk images.
3. Implement or carefully extend a dataset builder that combines these 28 pairs with the unmodified
   original dataset at `/Users/pot/Documents/大创/实验归档/origami-lora-pairs-v1-23` into a new
   versioned 51-pair ImageFolder directory. Never overwrite the V1 dataset.
4. Write one metadata row per pair. Use a common strong instruction to naturalize the complete
   subject, then add tag-specific clauses naming all relevant regions (hair/headwear, beard,
   wrinkles, scalp/ears/neck, clothing/shoulders/bust/pedestal). Preserve identity, age, skin tone,
   pose, gaze, expression, crop, garment layout, background, palette, and lighting.
5. Add focused tests for selection, exclusion, count, source/target pairing, portable paths, and
   metadata generation. Run the new script's `--help`, Ruff, relevant pytest, and
   `git diff --check`.
6. Build the real 51-pair dataset locally and make a simple contact sheet to catch path reversal,
   duplicate, crop, or pairing errors. Package it for AutoDL and report its upload source and exact
   server destination. Do not add unnecessary formal archive ceremony.
7. Prepare one directly copyable AutoDL command block, but do not run it until the operator says the
   instance is online. It must use `screen`, not `tmux`, and train a fresh rank-16 adapter from the
   base model with learning rate `1e-4`, effective batch 4, max 200 steps, and checkpoints at
   50/100/150/200. Do not resume checkpoint 100.
8. After the operator completes training, repeat the unchanged six-holdout seed-42 comparison for
   Base, frozen V1 checkpoint 100, and V2 checkpoints 50/100/150/200. Compare residual paper and
   identity/composition drift honestly; the useful target is at least 4/6 passes and at least one
   rescue among `002/011/018`, but do not hide a mixed result.

## Boundaries

- local macOS is for image generation, curation, packaging, code, and review; it cannot run CUDA;
- AutoDL uses `screen`, not `tmux`;
- for a GitHub operation on AutoDL, run `source /etc/network_turbo`, perform the single explicit Git
  fetch/push operation, then unset `http_proxy`, `https_proxy`, `HTTP_PROXY`, `HTTPS_PROXY`,
  `all_proxy`, and `ALL_PROXY`; use `git fetch origin main` plus
  `git merge --ff-only origin/main`, not a multi-branch `git pull`;
- do not reinstall or upgrade the prepared Torch/Diffusers/Accelerate environments;
- do not train on the six holdouts, rejected teacher outputs, or V2 IDs `021` and `023`;
- do not increase rank, steps, LoRA scale, seeds, models, or add ControlNet before the V2 data test;
- do not mix 3D, Clay, Needle-felt, or Origami into one adapter;
- do not restart formal-v1 or claim that 3/6 is a solved result;
- preserve all existing user changes, use versioned directories, run relevant tests and
  `git diff --check`, and push only with current authorization.
