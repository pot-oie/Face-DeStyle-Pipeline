# Multistyle routing and project-closure handoff

## Active decision

The next phase is not another LoRA campaign. The project already has enough evidence to close the
style-specific adaptation branch and consolidate a practical multistyle routing story.

Do not automatically generate another training bank, rent a GPU, resume an adapter, or repeat the
Origami workflow for Clay, 3D cartoon, or Needle-felt. Training time was not the only cost: strict
teacher-pair construction and full-frame curation were the dominant bottlenecks.

The active local task is to organize the completed evidence, make the per-style routing decision
explicit, and prepare a compact final research narrative and visual index. AutoDL has no prescribed
next run.

## Evidence that fixes this decision

### Painting-like styles

Original BF16 FLUX Kontext already produced a strong signal on comic, ink, and watercolor. In the
historical 40-image calibration review, FLUX passed 37/40 overall while each SDXL baseline passed
only 1--3/40. These styles do not justify separate LoRAs in this project.

### Needle-felt

The material-extension pilot found that a single Stage 1 edit usually removed most fiber cues and a
true second edit added little. Needle-felt should remain a prompt-only success/control unless a new
source set later contradicts that result. Do not train a Needle-felt LoRA.

### 3D cartoon

The existing eight-pair rank-16/200-step smoke adapter loaded and changed inference, but generally
made the editor more conservative and closer to the original 3D input. It did not reliably remove
large eyes, exaggerated geometry, CGI material, or synthetic lighting. Sequential open-model edits
also yielded no strict target in the reviewed probe, and only one strict closed-teacher target is
available. Preserve this as a useful negative adaptation result. Do not resume the adapter or start
the abandoned style-contrast19 run.

### Clay

Stage 1 usually retained clay. Stage 2 sometimes removed facial material but either left clay on
clothing or invented a different person. Only one strict closed-teacher pair is currently available.
Clay is therefore a routing/fallback problem, not a trainable LoRA dataset at present.

If the operator later explicitly requests one additional adaptation experiment, Clay is the only
reasonable first candidate. Before building a dataset, run at most a 6--8-source closed-teacher
feasibility pilot. Continue only if about five targets preserve the complete identity and
composition while removing clay across the full frame. Require roughly 20 strict pairs before
training. Otherwise stop. Do not begin this pilot under the current handoff.

### Origami

The first 23-pair rank-16 adapter improved the fixed six-holdout review from Base at about 1/6 to
3/6 at checkpoint 100. It remains the selected limited adapter.

The 51-pair V2 run and the CLIP-safe caption-only V2.1 run both completed fresh 200-step training
with checkpoints 50/100/150/200. Neither improved generalization: V2 and V2.1 reached at most 2/6
under the shared holdout prompt, and neither rescued the full set of hard material regions. A
source-specific diagnostic rescued `011`, showing that instruction alignment matters, but the
effective result remained around 3/6. Do not promote or continue V2/V2.1, increase rank or steps,
or perform another caption sweep.

The active Origami fallback for hard cases is a true residual-region second edit, not another
single-pass LoRA retrain.

## Practical style routing

| Style | Default path | Failure fallback | LoRA decision |
|---|---|---|---|
| Comic | Base FLUX prompt-only | record/review isolated failure | none |
| Ink | Base FLUX prompt-only | record/review isolated failure | none |
| Watercolor | Base FLUX prompt-only | record/review isolated failure | none |
| Needle-felt | Base FLUX Stage 1 | targeted second edit only when fiber remains | none |
| Origami | Base or frozen V1 checkpoint 100, described honestly as limited | residual-region second edit or teacher fallback | freeze V1; stop V2/V2.1 |
| Clay | Base Stage 1, then true Stage 2 for residual material | closed teacher or explicit failure | not enough pairs |
| 3D cartoon | Base FLUX as the reproducible baseline | closed teacher or explicit failure | freeze negative smoke |

This table is a research routing summary, not a claim that every source in a listed style will pass.

## Next local deliverable

Use only existing records and visual artifacts to create one concise closure package:

1. a factual multistyle result report that distinguishes completed real-model evidence from proposed
   fallbacks;
2. a compact visual index pointing to the best existing comparison/contact sheets without copying
   bulk images into Git;
3. a final per-style routing table consistent with this handoff;
4. an updated top-level handoff/index so a future reader does not mistake old Origami V2 or 3D
   style-contrast plans for active work;
5. a short list of optional future extensions, clearly labeled optional and requiring new operator
   authorization.

Prefer updating documentation and lightweight configuration over implementing a large orchestration
system. Do not rerun formal-v1 scoring, add metric suites, rebuild archives, or create acceptance
ceremony merely to close the project.

## Primary evidence files

- `docs/HANDOFF_MULTISTYLE_PAIR_BANK_AND_LORA.md`
- `docs/results/multistyle_pair_bank_stage2_review_20260824.md`
- `docs/results/origami_lora_holdout_review_20260824.md`
- `docs/results/origami_lora_v2_holdout_review_20260825.md`
- `docs/results/origami_lora_v21_holdout_review_20260826.md`
- `/Users/pot/Documents/大创/实验归档/showcase-20260825`
- `/Users/pot/Documents/大创/实验归档/returned-runs`

## Boundaries

- no new LoRA training is currently required;
- no AutoDL command should be issued without a new explicit operator request;
- preserve V1 Origami checkpoint 100 and the negative 3D/V2/V2.1 artifacts;
- do not call the selected Origami adapter a solved result: it passed about 3/6 fixed holdouts;
- do not call V2/V2.1 improvements: both failed to beat V1;
- do not claim closed-teacher outputs are ground-truth natural appearances;
- do not mix style-specific adapters into a multistyle LoRA merely to add another experiment;
- keep bulk images, checkpoints, returned runs, and source data out of Git.
