# Prompt for the next main implementation window

Continue `Face-DeStyle-Pipeline` from the completed Origami pair-bank handoff. Do not restart
formal-v1 work, regenerate the Origami targets, or resume the historical eight-pair 3D LoRA.

Local repository: `/Users/pot/Github/Face-DeStyle-Pipeline`

Local dataset: `/Users/pot/Documents/大创/Face-DeStyle-Data`

AutoDL project root: `/root/autodl-tmp/face-destyle`

First read completely:

1. `AGENTS.md`
2. `docs/HANDOFF_MULTISTYLE_PAIR_BANK_AND_LORA.md`
3. `docs/AUTODL_ORIGAMI_LORA_PREP.md`
4. `docs/results/multistyle_pair_bank_stage2_review_20260824.md`
5. `scripts/build_pair_bank_lora_dataset.py`
6. `data/manifests/multistyle-pair-bank/origami_target_selection_v1.csv`

Current state (2026-08-24):

- Base FLUX Stage 1 and true sequential Stage 2 were completed and reviewed for 3D, Clay, and
  Origami.
- A face-only review mistake accepted 19 Origami outputs with residual paper hair/clothing/bust.
  The resulting `origami-lora-pairs-v1-20` is withdrawn and must never be trained.
- Closed-teacher reconstruction and strict full-frame review are complete for all 24 Origami
  candidate sources. Twenty-three passed. `matv2-origami-004` failed three times because geometric
  skin/freckle marks remained and is excluded.
- The approved upload-ready dataset is
  `/Users/pot/Documents/大创/实验归档/origami-lora-pairs-v1-23.zip`.
- Upload it to
  `/root/autodl-tmp/face-destyle/packages/origami-lora-pairs-v1-23.zip`.
- The exact extraction, environment discovery, and fresh rank-16 300-step Kontext LoRA command are
  frozen in `docs/AUTODL_ORIGAMI_LORA_PREP.md`.
- 3D and Clay still have only one strict pair each and are not approved for LoRA training.

Your first concrete task is to follow `docs/AUTODL_ORIGAMI_LORA_PREP.md` exactly on AutoDL. Use
`source /etc/network_turbo` only around `git fetch origin main`, then unset all proxy variables.
Locate the already prepared training environment; do not reinstall or upgrade Torch, Diffusers,
Transformers, Accelerate, or the external trainer checkout. Train from scratch into
`outputs/origami-destyle-lora-teacher23-r16-steps300`, preserving checkpoints 100, 200, and 300.

After training succeeds, the next task is a fixed held-out comparison of Base versus checkpoints
100/200/300 on the six Origami holdouts. Select the least overfit useful checkpoint from images,
not training loss. Do not add seeds, prompt sweeps, ControlNet, pose, depth, a new base model, or a
mixed-style LoRA before that comparison.

Preserve all existing user changes, inspect the worktree before editing, run relevant tests and
`git diff --check`, and report counts honestly. Local macOS is for curation and code only; CUDA and
FLUX execution stay on AutoDL.
