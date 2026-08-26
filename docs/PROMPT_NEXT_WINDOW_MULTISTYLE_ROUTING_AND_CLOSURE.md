# Prompt for the next multistyle routing and closure window

> Historical prompt, fully superseded and completed on 2026-08-26. The missing true-sequential
> evidence, Origami residual probe, executable human-review router, and final closure report now
> exist. Do not repeat any task or launch any command from this prompt.

Continue `Face-DeStyle-Pipeline` from the completed Origami V1/V2/V2.1 experiments and the earlier
3D, Clay, Needle-felt, comic, ink, and watercolor investigations.

Local repository: `/Users/pot/Github/Face-DeStyle-Pipeline`

Local experiment archive: `/Users/pot/Documents/大创/实验归档`

First read completely:

1. `AGENTS.md`
2. `docs/HANDOFF_MULTISTYLE_ROUTING_AND_PROJECT_CLOSURE.md`
3. `docs/HANDOFF_MULTISTYLE_PAIR_BANK_AND_LORA.md`
4. `docs/results/multistyle_pair_bank_stage2_review_20260824.md`
5. `docs/results/origami_lora_holdout_review_20260824.md`
6. `docs/results/origami_lora_v2_holdout_review_20260825.md`
7. `docs/results/origami_lora_v21_holdout_review_20260826.md`

## Fixed interpretation

- Do not train another LoRA in this window.
- Comic, Ink, and Watercolor use Base FLUX prompt-only; no style LoRA is planned.
- Needle-felt remains a prompt-only success/control; do not train it.
- The eight-pair 3D LoRA is a negative smoke result and must not be resumed. The abandoned
  style-contrast19 plan is not active.
- Clay has only one strict teacher pair and is not ready for LoRA training.
- Origami V1 checkpoint 100 remains the selected limited adapter at about 3/6 strict holdout passes.
- Origami V2 and V2.1 did not beat V1 and must not be promoted, resumed, or retuned.
- AutoDL has no active prescribed run. Do not issue remote commands unless the operator creates a
  new explicit GPU task.

## Concrete task

Close and organize the multistyle adaptation branch using existing evidence only.

1. Inspect the current working tree and preserve unrelated user changes.
2. Review the listed reports and the compact visual evidence already under
   `/Users/pot/Documents/大创/实验归档/showcase-20260825`. Do not reopen bulk archives unless a
   specific factual gap requires it.
3. Create one concise final multistyle result report. It must state, by style, what was actually run,
   what succeeded or failed, what route is recommended, and the limitation of the evidence.
4. Create or update a lightweight project index that points to the relevant reports and local visual
   evidence. Do not commit raw images or weights.
5. Make the practical routing explicit:
   - Comic/Ink/Watercolor: Base FLUX prompt-only;
   - Needle-felt: Base Stage 1, targeted second edit only for residual fiber;
   - Origami: frozen V1 checkpoint 100 as an optional limited adapter, then residual-region second
     edit or teacher fallback for hard cases;
   - Clay: Stage 1 -> true Stage 2 -> teacher/failure routing, with no LoRA yet;
   - 3D cartoon: Base/teacher/failure routing, preserving the negative LoRA smoke result.
6. Reconcile stale “next step” text in active handoff/index documents so no reader is instructed to
   launch Origami V2/V2.1 or 3D style-contrast training.
7. Run `git diff --check` and any narrowly relevant documentation/link checks. Do not run heavy
   model tests for documentation-only changes.
8. Commit and push the completed closure update when the working tree contains only intended files.

## Optional future work, not authorized now

If the operator later asks for one more adaptation experiment, Clay is the first candidate, but only
after a 6--8-source closed-teacher feasibility pilot and only if roughly five full-frame targets pass.
Require about 20 strict pairs before LoRA training. This is a conditional future option, not part of
the current task.

Do not add new datasets, generate images, start AutoDL, train a selector, create a multistyle LoRA,
or restart formal evaluation in this window. The desired output is a clean, evidence-backed project
closure and routing handoff, not another experiment.
