# Prompt for the next main implementation window

Continue `Face-DeStyle-Pipeline` from the completed first Origami LoRA experiment and the targeted
hard-pair expansion handoff. The operator has temporarily shut down AutoDL. Do not attempt remote
commands until the operator explicitly reports that the instance is running again.

Local repository: `/Users/pot/Github/Face-DeStyle-Pipeline`

Local dataset: `/Users/pot/Documents/大创/Face-DeStyle-Data`

AutoDL project root when online: `/root/autodl-tmp/face-destyle`

First read completely:

1. `AGENTS.md`
2. `docs/HANDOFF_MULTISTYLE_PAIR_BANK_AND_LORA.md`
3. `docs/PROMPT_ORIGAMI_HARD_PAIR_EXPANSION_V2.md`
4. `docs/results/multistyle_pair_bank_stage2_review_20260824.md`
5. `docs/results/origami_lora_holdout_review_20260824.md`
6. `scripts/build_pair_bank_lora_dataset.py`
7. `scripts/run_origami_lora_holdout_eval.sh`
8. `data/manifests/multistyle-pair-bank/origami_target_selection_v1.csv`

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

## Expected incoming artifact

A separate image-generation/curation window is responsible for
`docs/PROMPT_ORIGAMI_HARD_PAIR_EXPANSION_V2.md`. It should return:

`/Users/pot/Desktop/origami-hard-pairs-v2-generation-and-review.zip`

containing 36 new hard Origami source candidates, accepted/rejected teacher outputs, manifests,
prompt notes, and contact sheets under an `origami_hard_pairs_v2` tree. The target is 24--30 strict
accepted pairs. These images are untrusted until this main window independently inspects them.

## First concrete task after the artifact arrives

1. Inspect the ZIP without overwriting existing data; verify counts, paths, image readability, IDs,
   manifests, and absence of the six protected holdouts.
2. Build source/target comparison sheets and conduct strict full-frame review. Reject paper in
   hair/headwear/beard/scalp/neck/clothing/bust/pedestal and reject identity, age, pose, gaze, or
   composition drift.
3. Report accepted counts by difficulty category. Do not force the result to 24 if fewer pass.
4. Integrate only accepted pairs with the original 23-pair dataset. Preserve the original dataset
   and selection manifest; create versioned V2 manifests and a new output directory.
5. Replace the single generic caption with per-example instructions that explicitly name all
   material regions present in that source while preserving identity and garment structure.
6. Build and visually verify an approximately 47--53-pair ImageFolder package. Keep all six original
   holdouts out of training.
7. Prepare, but do not launch until the operator authorizes AutoDL, a fresh rank-16 Kontext LoRA
   run from the base model. Do not resume checkpoint 100. Keep learning rate `1e-4`, effective batch
   4, and choose steps by exposure: for roughly 48 pairs, evaluate checkpoints 50/100/150/200 and
   stop at 200 rather than automatically running 300.
8. After training, repeat the exact six-holdout seed-42 comparison. The V2 gate is at least 4/6
   strict passes, at least one rescue among `002/011/018`, and no worse identity drift than the
   frozen checkpoint-100 baseline.

## Boundaries

- local macOS is for image generation, curation, packaging, code, and review; it cannot run CUDA;
- AutoDL uses `screen`, not `tmux`;
- GitHub operations on AutoDL use `source /etc/network_turbo`, followed immediately by unsetting all
  proxy variables;
- do not reinstall or upgrade the prepared Torch/Diffusers/Accelerate environments;
- do not train on the six holdouts or on rejected teacher outputs;
- do not increase rank, steps, LoRA scale, seeds, models, or add ControlNet before the V2 data test;
- do not mix 3D, Clay, Needle-felt, or Origami into one adapter;
- do not restart formal-v1 or claim that 3/6 is a solved result;
- preserve all existing user changes, use versioned directories, run relevant tests and
  `git diff --check`, and push only with current authorization.
