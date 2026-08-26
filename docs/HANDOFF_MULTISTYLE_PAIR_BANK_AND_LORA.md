# Multistyle reconstruction pair bank and targeted LoRA handoff

> Closure update (2026-08-26): this file preserves the pair-bank and training history. Active work
> has moved to `docs/HANDOFF_MULTISTYLE_ROUTING_AND_PROJECT_CLOSURE.md`. Do not infer a new LoRA run
> from older plans below.

## Status and authority

This handoff records the active research direction as of 2026-08-24. It supersedes the immediate
plan to keep tuning or retraining the eight-pair 3D LoRA. Older formal-v1 and material-extension-v1
documents remain useful history, but their freezes, blind review, acceptance gates, hashing, and
prescribed next steps are not active requirements. Follow the operator policy at the top of
`AGENTS.md`: keep the work lightweight, visually interpretable, and useful for the research story.

At that stage, the goal shifted away from rescuing one tiny LoRA run and toward a repeatable
small-scale research sequence:

1. identify style classes that a strong editor still fails to remove;
2. collect a larger bank of genuine styled portrait inputs;
3. generate several natural-reconstruction candidates per input with open and, where useful,
   closed-source teacher models;
4. visually select reliable styled-input/natural-target pairs;
5. train a style-specific FLUX Kontext LoRA only after enough useful pairs exist;
6. compare Base FLUX and LoRA on styled inputs not used for training.

Do not restart formal-v1 analysis, blind testing, repeat rating, archive hashing, or large metric
suites unless the operator explicitly asks for them.

## Research relationship and claim boundary

The senior student's paper is *Learning to Stylize by Learning to Destylize*, available locally at
`/Users/pot/Documents/大创/2509.05970v2.pdf`. Its important ideas for this project are:

- reverse the usual synthetic-data direction so that high-quality artistic images remain useful
  supervision while natural reconstructions are generated and filtered;
- use progressive generation and quality filtering rather than trusting every model output;
- build training pairs or triplets only from candidates that retain the intended content;
- use an image-to-image LoRA as a later optimization stage, not as a substitute for constructing
  the data correctly.

The paper works at much greater scale and depth. This repository is a small face-domain exploratory
subproject that initially helped explore the direction. It must not claim the paper's dataset,
method names, benchmark results, scale, or novelty. A good interview-level claim is that the project
reproduced the reasoning pattern on a compact domain, found a specific failure boundary, and tested
a failure-targeted adaptation path.

## Research narrative established so far

The project began with SDXL prompt-only and Canny variants. Those methods generally preserved
stylized contours and did not reliably produce natural portraits. Original-BF16
`FLUX.1-Kontext-dev` was much stronger on comic, ink, and watercolor sources. Historical
calibration review found 37/40 FLUX candidates passing the project's strict human rule, compared
with only 1--3/40 for each SDXL baseline. The 3D category was weaker at 8/10 and repeatedly retained
rendered geometry or material cues.

The material-extension pilot then examined 3D cartoon, clay, and needle felt:

- needle felt: Stage 1 usually removed most fiber cues; Stage 2 added little;
- clay: Stage 1 mostly failed; the stronger Stage 2 often removed the clay material but could invent
  human identity or facial structure;
- 3D cartoon: the two prompt variants were very similar and retained large eyes, exaggerated
  geometry, plastic/CGI material, or synthetic lighting.

This supports a coherent failure boundary: painting-like surface styles are often handled by a
strong pretrained editor, while geometry/material-entangled styles remain difficult. The next work
should focus on 3D, clay, and origami rather than training a LoRA for every style. Needle felt can
remain a prompt-only success/control unless a larger source set disproves that observation.

## Existing material inventory

Operator update (2026-08-23): the hard-style source banks are still being filled. The counts below
are a snapshot, not a final curation target, and the 67 raw 3D-category files include non-portrait
animation frames and other unusable candidates. Do not freeze a final selected list yet. The
source-independent generation and review tooling is documented in
`docs/MULTISTYLE_PAIR_BANK_WORKFLOW.md` and can be completed while acquisition continues.

Later operator update (2026-08-23): source generation is complete for the current pair-bank round.
The new material-v2 bank contains 24 Clay, 12 replacement Needle-felt, and 30 Origami portraits.
Visual curation established lightweight source lists under
`data/manifests/multistyle-pair-bank/`: 27 candidate plus 6 holdout 3D portraits, 19 candidate plus
5 holdout new Clay portraits, and 24 candidate plus 6 holdout Origami portraits. Historical museum
Clay fragments and non-portrait 3D animation frames remain listed as rejected rather than silently
counted as usable portraits.

The local dataset root is `/Users/pot/Documents/大创/Face-DeStyle-Data`. A simple raw-file count on
2026-08-23 found:

| Style | Raw files | Interpretation |
|---|---:|---|
| 3D cartoon | 67 | enough candidates to curate a 40--50 image source bank |
| Clay | 12 | too small; expand to roughly 30--40 |
| Needle felt | 12 | enough for a baseline, but can expand to roughly 24 |
| Comic | 60 | baseline already comparatively strong |
| Ink | 61 | baseline already comparatively strong |
| Watercolor | 92 | baseline already comparatively strong |
| Origami | 0 | create roughly 30 new source images |

These are raw counts, not accepted unique counts. Some 3D files are extra candidate frames or may be
near duplicates. Inspect them visually and choose diverse, usable portraits. Do not turn this into
a cryptographic audit; a selected-file list and contact sheet are sufficient.

## Completed 3D LoRA smoke work

The repository implements a small synthetic-pair path:

- `scripts/build_3d_lora_smoke_pairs.py` creates natural portrait targets, 3D conditions, and
  ImageFolder metadata;
- the FLUX Kontext backend and `scripts/run_flux_kontext_probe.py` can load a local LoRA with
  `--lora-weights` and `--lora-scale`;
- `configs/styles_3d_lora.yaml` contains the stronger prompt shared by LoRA training metadata and
  future Base/LoRA comparison runs.

Important commits include:

- `1c24863 Add 3D destylization LoRA smoke path`;
- `1118752 Release GPU memory between pair stages`;
- `e695904 Refine restrained 3D pair preview`;
- `7c5901c Use restrained V2 prompt for full 3D pair run`;
- `6438679 Align 3D LoRA training and evaluation prompts`.

Twenty-four synthetic natural-target/3D-condition pairs were generated with the restrained V2
prompt. The returned archive is
`/Users/pot/Desktop/3d-lora-smoke-pairs-v2-full.zip`. Visual review showed that many conditions had
oversized eyes, waxy skin, or facial drift. Eight conservative pairs were selected:
`004, 005, 009, 010, 016, 019, 021, 023`.

The eight-pair rank-16 LoRA completed 200 steps on AutoDL and was tested on the five original 3D
pilot sources. The returned comparison archive is
`/Users/pot/Desktop/3d-lora-base-vs-adapted-pilot5.zip`. Exact three-column comparison sheets are at
`/Users/pot/Documents/大创/实验归档/3d-lora-base-vs-adapted-pilot5-comparison`.

The result is a useful negative smoke finding:

- the LoRA outputs differed from Base FLUX, proving the adapter loaded and affected inference;
- the changes were not a reliable destylization improvement;
- the LoRA generally became more conservative and closer to the original 3D input;
- on examples 002 and 005 it reduced some of Base FLUX's realistic texture;
- it did not consistently reduce large eyes, 3D geometry, or CGI material.

Do not present this LoRA as an optimization. It is evidence that selecting only low-drift,
low-contrast synthetic pairs can teach near-copy behavior.

## Paused style-contrast19 attempt

A second synthetic subset was proposed by excluding grayscale targets `006, 007, 011, 017, 018`
and retaining the other 19 pairs. It was intended to add stronger style contrast. The AutoDL
dataset directory may already exist at
`/root/autodl-tmp/face-destyle/data/3d-lora-style-contrast19`, but this has not been confirmed.

Training did not start: the shell returned `bash: accelerate: command not found` and exit code 127.
That means the active shell did not expose the prepared training environment. It is not a CUDA,
model, dataset, or training result. Do not spend time fixing this now and do not resume the
style-contrast19 plan automatically. The operator correctly identified that it still relies too
heavily on lucky synthetic stylization and does not solve the data problem.

The first eight-pair LoRA remains on AutoDL under approximately
`/root/autodl-tmp/face-destyle/outputs/3d-destyle-lora-smoke-r16-steps200`. Preserve it as the
conservative negative baseline. Do not overwrite or continue its checkpoint.

## Historical data-construction strategy

### 1. Curate genuine styled source banks

Start with the existing 67 3D files. Build a contact sheet and select roughly 40--50 diverse,
clearly 3D portraits. Reserve 5--10 of them for later qualitative Base/LoRA comparison. Selection
should favor a range of eye exaggeration, facial geometry, skin/material rendering, pose, age,
lighting, and background complexity.

The separate material-generation task should expand clay to roughly 30--40 total source images,
needle felt to roughly 24, and origami to roughly 30. New sources should depict fictional people,
one primary face, obvious style, no text/logo/known character, and 1024-square output when possible.

### 2. Generate multiple natural candidates from each styled source

For each source artwork, create a small candidate set:

1. one-pass Base FLUX reconstruction;
2. a true sequential second edit that uses the Stage 1 output as the Stage 2 input;
3. one closed-source teacher reconstruction where available.

The old material-extension `--prompt-stage stage2` run edited the original source again; it was not
a true Stage 1-to-Stage 2 chain. New tooling should make the sequential input explicit, keep outputs
in separate directories, and avoid silently confusing those two designs.

Closed-source outputs are allowed as private-research teacher targets. Keep a simple note of the
teacher model and prompt. Do not claim that the final local LoRA is itself the closed model.

### 3. Select reliable pairs visually

Create review sheets with columns:

`styled source | FLUX Stage 1 | FLUX Stage 1->2 | closed teacher`

For each source, select at most one target. Accept a pair when the candidate is meaningfully more
natural while retaining facial evidence, pose, expression, clothing, composition, and important
background structure. Reject the source when every candidate invents a new person, destroys the
composition, or remains strongly stylized. No blind review or formal score threshold is required.

The intended scale is roughly 30--50 styled sources per hard style, 2--3 candidates per source, and
20--40 selected training pairs. More source images are useful only when the corresponding natural
targets are good.

### 4. Train style-specific LoRAs

Train 3D, clay, and origami separately at first. Their failure mechanisms differ, and a tiny mixed
dataset may dilute the learning signal. Do not train a single multistyle adapter until each style
has enough accepted pairs and the style-specific experiments are understood.

Use held-out genuine styled sources for a simple three-column comparison:

`original styled input | Base FLUX | adapted FLUX`

Judge visible changes in style/material removal, face/content preservation, and artifacts. Do not
increase steps or LoRA scale merely because outputs differ.

### 5. Add a lightweight selector only after candidates exist

A useful shared selector has three concepts:

- naturalness/style removal;
- content and composition preservation;
- face consistency and artifact quality.

Begin with human selection and optional VLM suggestions. After roughly 100--200 candidate decisions
exist, a lightweight ranking model or structured VLM scorer can be studied. Do not train a scorer
before there are meaningful labels, and do not collapse the three concepts into one opaque score.

## Chronology of the completed implementation windows

Update (2026-08-24): all three curated Stage 1 batches completed on AutoDL and were reviewed
locally: 27 3D, 19 Clay, and 24 Origami outputs, all with zero runner failures. Fourteen Origami
outputs are provisional Stage 1 target candidates. No 3D or Clay Stage 1 output is yet reliable
enough to count as a selected training target. The reviewed routing table is
`docs/results/multistyle_pair_bank_stage1_routing_20260824.csv`; the exact next AutoDL commands are
in `docs/AUTODL_MULTISTYLE_PAIR_BANK_STAGE2_SUBSETS.md`. Run only those 8/12/10 sequential subsets
before the next visual review. Do not interpret the 14 provisional Origami candidates as a frozen
training set and do not start LoRA yet.

Later update (2026-08-24): the reviewed sequential subsets also completed at 8/8 3D, 12/12 Clay,
and 10/10 Origami with zero runner failures. The visual result is recorded in
`docs/results/multistyle_pair_bank_stage2_review_20260824.md`. No 3D target passed. Two Clay Stage 2
targets passed, while two other Clay outputs were rejected for identity drift. At that stage,
Origami had 19 apparently available open-model targets: 14 from Stage 1 and five from Stage 2. The
next action was a closed-teacher pilot rather than another sequential FLUX pass. The later
full-frame review below supersedes that provisional count.

Closed-teacher completion update (2026-08-24): OpenAI built-in reference-image edits were generated
through Codex and retained for one 3D, one Clay, and 23 Origami sources. Prompts and outputs remain
in the local experiment archive. A first face-only review incorrectly counted 20 Origami targets
even though 19 retained obvious folded-paper hair, headwear, clothing, or bust material; that
dataset is withdrawn and marked `DO_NOT_TRAIN.md`. The replacement pass reviewed the complete
portrait and accepted 23 of 24 candidate sources. `matv2-origami-004` was rejected after three
attempts retained geometric skin/freckle artifacts. The frozen ready-to-train artifact is
`/Users/pot/Documents/大创/实验归档/origami-lora-pairs-v1-23.zip`; its AutoDL upload and fresh
300-step training command are in `docs/AUTODL_ORIGAMI_LORA_PREP.md`. 3D and Clay still have one
strict target each. Never train the rejected 20-pair Origami set.

Origami LoRA training and held-out evaluation update (2026-08-24): the approved 23-pair dataset
trained successfully from scratch with rank 16, effective batch 4, learning rate `1e-4`, and 300
maximum steps. Checkpoints 100, 200, and 300 plus the final weight were saved. The fixed six-source
holdout comparison ran Base FLUX and all three checkpoints at seed 42, 28 steps, guidance 2.5, and
LoRA scale 1.0 with 6/6 outputs and zero failures per method. Full-frame review found roughly 1/6
strict pass for Base and 3/6 for every LoRA checkpoint. Checkpoint 100 is the selected adapter:
later checkpoints did not rescue more cases and progressively changed face shape, age, or
expression. Its AutoDL path and SHA-256 are:

```text
/root/autodl-tmp/face-destyle/outputs/origami-destyle-lora-teacher23-r16-steps300/checkpoint-100/pytorch_lora_weights.safetensors
06ab9433e341713aaaa0edb11849db5e687b47ad0ca930c121cea49277eca7c4
```

The six holdouts remain protected evaluation data. `007`, `023`, and `030` passed under checkpoint
100; `011` and `018` improved only partially because paper scalp/hair/clothing remained; `002`
remained a strong failure with paper hair, clothing, and bust. Do not train on any of these six.
The detailed review is `docs/results/origami_lora_holdout_review_20260824.md`.
The next intervention is a targeted difficult-pair expansion, not more steps, a higher rank, or a
continuation from checkpoint 100. The generation/curation task is specified in
`docs/PROMPT_ORIGAMI_HARD_PAIR_EXPANSION_V2.md`.

1. Preserve checkpoint 100 as the first positive but limited Origami baseline; do not continue it.
2. Keep the original six Origami holdouts sealed and unchanged.
3. Generate 36 new difficult fictional Origami sources and identity-preserving teacher targets
   using `docs/PROMPT_ORIGAMI_HARD_PAIR_EXPANSION_V2.md`.
4. Strictly retain about 24--30 new full-frame pairs, concentrating on paper hair/headwear, elderly
   beards, severe scalp/neck planes, and clothing/bust/pedestal material.
5. Independently inspect the returned package before merging it with the original 23 pairs.
6. Use stronger per-example material-region instructions and build an approximately 47--53-pair
   V2 dataset.
7. Train a new rank-16 adapter from the base Kontext model, not from checkpoint 100. Preserve early
   checkpoints and target roughly the same 17 effective epochs that produced the current optimum.
8. Re-run the unchanged six-holdout comparison. Require at least 4/6 strict passes and rescue at
   least one of `002`, `011`, or `018` without worse identity drift.

Origami difficult-pair expansion update (2026-08-25): the local generation window produced 36 new
sources and 30 generator-accepted teachers. Independent full-frame review retained 28. It excluded
`origami-hard-v2-021` and `origami-hard-v2-023` because their visible bust/pedestal support was
removed rather than converted while preserving composition. The retained set has broad coverage of
hair/headwear, elderly wrinkles and beards, bald scalp/neck planes, clothing/bust material, profile,
dark-skin, and strong-shadow cases. Combined with the original 23 strict pairs, the next training
dataset is exactly 51 pairs. This is enough for the next experiment; do not generate more before
testing it. Package inventory, accepted IDs, and the review rationale are recorded in
`docs/results/origami_hard_pairs_v2_review_20260825.md`.

The subsequent local task created a repository-owned V2 selection manifest, normalized the
delivered absolute CSV paths to portable source-relative metadata, built and visually checked the
51-pair ImageFolder package, and prepared the fresh rank-16/200-step run. It did not modify the
original 23-pair dataset or train directly from the generation window's 30-row
`selected_pairs.csv`.

Origami V2 result update (2026-08-25): the exact 51-pair dataset trained successfully from Base for
200 steps and saved checkpoints 50/100/150/200. The unchanged six-holdout comparison completed for
Base, frozen V1 checkpoint 100, and all four V2 checkpoints. Strict review found Base 1/6, V1
checkpoint 100 at 3/6, V2 checkpoint 50 at 0/6, and V2 checkpoints 100/150/200 at 2/6 each. V2-200
removed paper strongly on `030` but changed beard color, apparent age, and expression, so it does
not receive a strict pass. None of `002`, `011`, or `018` was rescued. V1 checkpoint 100 remains
the selected adapter; do not continue V2 automatically or call it an improvement. Full review:
`docs/results/origami_lora_v2_holdout_review_20260825.md`.

Prompt-alignment diagnostic update (2026-08-25): V2 checkpoint 200 was rerun only on unresolved
holdouts `002`, `011`, and `018` with source-specific full-subject instructions while seed 42,
28 steps, guidance 2.5, and LoRA scale 1.0 remained frozen. `011` was rescued strictly. `002`
naturalized the face, hair, and beard but retained paper clothing, bust, and pedestal; `018`
naturalized the face but retained paper outer hair/headwear and clothing. This supports a partial
train/evaluation instruction mismatch but also confirms a remaining full-subject generalization
limit. The effective strict result is still only about 3/6, not the required 4/6. Do not promote
V2 or start a parameter sweep. V1 checkpoint 100 remains selected. The returned archives are now
under `/Users/pot/Documents/大创/实验归档/returned-runs`, and compact visual evidence is under
`/Users/pot/Documents/大创/实验归档/showcase-20260825`.

V2.1 caption-only update (2026-08-26): the same 51 visual pairs were retrained from Base with five
CLIP-safe templates distributed `11/10/10/10/10`. The run completed all 200 steps and saved valid
checkpoints 50/100/150/200 plus final weights. Fixed six-holdout review found checkpoint 50 at 0/6
and checkpoints 100/150/200 at 2/6 (`007`, `023`). `002` and `018` still retain paper hair,
clothing, and support regions; `011` invents hair on the bald subject and retains paper clothing;
`030` removes paper but changes beard color, age, and expression. Caption compression did not
improve V2. V1 checkpoint 100 remains selected, and no V2.1 checkpoint should be promoted or
continued. Full review: `docs/results/origami_lora_v21_holdout_review_20260826.md`.

## Environment notes

Local macOS is for curation, contact sheets, code, Ruff, and pytest. It cannot run CUDA or FLUX.
AutoDL project root is `/root/autodl-tmp/face-destyle`; repository is normally
`/root/autodl-tmp/face-destyle/code/Face-DeStyle-Pipeline` and the local FLUX model is
`/root/autodl-tmp/face-destyle/models/diffusion/FLUX.1-Kontext-dev`.

The external official Kontext trainer is expected under
`/root/autodl-tmp/face-destyle/code/diffusers-kontext-training/examples/dreambooth/`.
The command `accelerate` is environment-dependent. Locate and activate the already prepared
training environment before future training; do not reinstall or upgrade packages merely because a
new shell lacks the command.

## What not to do next

- do not continue the eight-pair checkpoint;
- do not continue the 23-pair Origami adapter past checkpoint 100;
- do not train on Origami holdouts `002`, `007`, `011`, `018`, `023`, or `030`;
- do not add easy or near-duplicate Origami pairs merely to increase the count;
- do not train style-contrast19 merely because its folder may exist;
- do not manufacture all conditions from natural portraits and hope prompt constraints preserve
  anatomy;
- do not mix 3D, clay, felt, and origami into one tiny LoRA;
- do not add seeds, prompt sweeps, ControlNet, pose, depth, or new base models before constructing
  the pair bank;
- do not claim that a visible LoRA difference is an optimization;
- do not reopen formal-v1 blind evaluation, hashing, or repeat-rating workflows.
