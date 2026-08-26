# Experiment and visual evidence index

## Current reading order

1. [`results/multistyle_project_closure_20260826.md`](results/multistyle_project_closure_20260826.md)
   — final evidence summary and per-style routing contract.
2. [`HANDOFF_MULTISTYLE_ROUTING_AND_PROJECT_CLOSURE.md`](HANDOFF_MULTISTYLE_ROUTING_AND_PROJECT_CLOSURE.md)
   — project boundary and closure handoff.
3. [`results/formal_v1_reduced_heldout_20260822.md`](results/formal_v1_reduced_heldout_20260822.md)
   — exploratory 32-source, five-method reduced replication.
4. [`HANDOFF_MULTISTYLE_PAIR_BANK_AND_LORA.md`](HANDOFF_MULTISTYLE_PAIR_BANK_AND_LORA.md)
   — chronological pair-bank and adaptation history; not an active training plan.

## Repository-owned result records

| Evidence | What it supports | Status |
|---|---|---|
| [`results/formal_v1_reduced_heldout_20260822.md`](results/formal_v1_reduced_heldout_20260822.md) and [`results/formal_v1_reduced_32/`](results/formal_v1_reduced_32/) | FLUX 22/32 overall; 6/8 for comic, ink, and watercolor and 4/8 for 3D cartoon; SDXL baselines at 0--1/32 | completed exploratory reduced replication; post-unblinding and completion-informed |
| [`results/multistyle_pair_bank_stage2_review_20260824.md`](results/multistyle_pair_bank_stage2_review_20260824.md) | true sequential Stage 2 probes, corrected full-frame review, 0/8 strict 3D and 0/12 strict Clay open-model targets, final 23-pair Origami teacher bank | completed visual review |
| [`results/origami_lora_holdout_review_20260824.md`](results/origami_lora_holdout_review_20260824.md) | V1 Base about 1/6 versus checkpoint 100 about 3/6; later checkpoints add drift without more passes | completed six-holdout review; selected limited adapter |
| [`results/origami_hard_pairs_v2_review_20260825.md`](results/origami_hard_pairs_v2_review_20260825.md) | 28 accepted hard pairs and the exact 51-pair V2 construction rationale | completed dataset review; historical input to finished V2 |
| [`results/origami_lora_v2_holdout_review_20260825.md`](results/origami_lora_v2_holdout_review_20260825.md) | V2 at most 2/6 under the shared prompt; prompt-alignment diagnostic rescues `011` only | completed negative/mixed result |
| [`results/origami_lora_v21_holdout_review_20260826.md`](results/origami_lora_v21_holdout_review_20260826.md) | caption-only V2.1 at most 2/6 and no improvement over V1 | completed negative result |
| [`HANDOFF_MATERIAL_STYLE_EXTENSION_V1.md`](HANDOFF_MATERIAL_STYLE_EXTENSION_V1.md) and [`HANDOFF_MULTISTYLE_PAIR_BANK_AND_LORA.md`](HANDOFF_MULTISTYLE_PAIR_BANK_AND_LORA.md) | five-source material pilot design and the qualitative Needle-felt, Clay, and 3D outcomes | historical experiment contract plus consolidated outcome record |

## Local visual evidence

The compact visual set is intentionally not tracked by Git:

```text
/Users/pot/Documents/大创/实验归档/showcase-20260825
```

These paths are machine-local evidence pointers. They are not redistribution links and may not
exist on another checkout.

| File | Comparison shown | Claim boundary |
|---|---|---|
| `01-3d-lora-original-base-adapted.jpg` | five 3D sources: input / Base / eight-pair LoRA | adapter changed inference but usually stayed close to the rendered input |
| `02-clay-source-stage1-stage2-teacher.jpg` | Clay source / Stage 1 / true Stage 2 / available teacher | open edits retain clay or leave missing candidates; one teacher example shows a possible fallback, not a universal result |
| `03-origami-source-stage1-teacher.jpg` | Origami source / Stage 1 / available teacher | facial naturalization can coexist with residual full-frame paper; teacher targets motivated strict full-frame review |
| `04-origami-v1-training-pairs-23.jpg` | the 23 strict V1 condition/teacher pairs | visual inventory of the selected V1 supervision |
| `05-origami-v2-training-pairs-51.jpg` | the complete 51-pair V2 supervision bank | more and harder reviewed pairs did not guarantee better holdout generalization |
| `06-origami-v1-v2-checkpoint-comparison.jpg` | six holdouts across source, Base, V1-100, and V2 checkpoints | V1-100 remains the best strict aggregate at about 3/6 |
| `07-origami-hard-002-checkpoints.jpg` | hard case `002` across checkpoints | paper hair, clothing, bust, and support persist |
| `08-origami-hard-011-checkpoints.jpg` | hard case `011` across checkpoints | shared-prompt checkpoints do not remove scalp/garment construction |
| `09-origami-hard-018-checkpoints.jpg` | hard case `018` across checkpoints | face improves while outer hair/headwear and garment remain paper |
| `10-origami-030-material-vs-identity-drift.jpg` | hard case `030` across checkpoints | stronger material removal conflicts with age, beard-color, and expression preservation |
| `11-origami-prompt-alignment-hard3.jpg` | V2-200 source-specific prompts on `002`, `011`, `018` | only `011` strictly passes; wording mismatch is contributory but not sufficient |
| `12-origami-v21-clip77-negative.jpg` | V2.1 checkpoints on the six fixed holdouts | shorter CLIP-safe captions do not solve full-subject residual material |

Full archives, source images, pair packages, logs, and checkpoint weights remain under the local
experiment archive or AutoDL data disk. They must not be copied into this repository. The selected
Origami V1 checkpoint 100 is identified in the V1 report by its server path and SHA-256; this index
does not duplicate the weight.

## Historical execution documents

Files named `AUTODL_*`, older `PROMPT_*`, and the detailed pair-bank workflow are preserved to show
how completed experiments were run. Unless a document explicitly links back to the current closure
report, its commands and “next experiment” language describe historical state. There is no active
AutoDL run, LoRA training, new dataset generation, or formal evaluation task.
