# Experiment and visual evidence index

## Current reading order

1. [`results/multistyle_project_closure_20260826.md`](results/multistyle_project_closure_20260826.md)
   — final evidence summary, strict outcomes, and per-style routing contract.
2. [`results/multistyle_routing_gap_v1_review_20260826.csv`](results/multistyle_routing_gap_v1_review_20260826.csv)
   — all 24 Comic/Ink/Watercolor/Needle-felt Stage 1/Stage 2 decisions.
3. [`results/origami_v1_residual_stage2_review_20260826.md`](results/origami_v1_residual_stage2_review_20260826.md)
   — final three-source V1-100 residual-edit review; 0/3 strict rescues.
4. [`HANDOFF_MULTISTYLE_PROCESSING_ROUTER.md`](HANDOFF_MULTISTYLE_PROCESSING_ROUTER.md)
   — completed run record and executable human-review router usage.
5. [`results/multistyle_processing_coverage_audit_20260826.md`](results/multistyle_processing_coverage_audit_20260826.md)
   — pre-completion coverage audit that identified the resolved gaps.
6. [`HANDOFF_MULTISTYLE_ROUTING_AND_PROJECT_CLOSURE.md`](HANDOFF_MULTISTYLE_ROUTING_AND_PROJECT_CLOSURE.md)
   — project boundary and closure handoff.
7. [`results/formal_v1_reduced_heldout_20260822.md`](results/formal_v1_reduced_heldout_20260822.md)
   — exploratory 32-source, five-method reduced replication.
8. [`HANDOFF_MULTISTYLE_PAIR_BANK_AND_LORA.md`](HANDOFF_MULTISTYLE_PAIR_BANK_AND_LORA.md)
   — chronological pair-bank and adaptation history; not an active training plan.

## Repository-owned result records

| Evidence | What it supports | Status |
|---|---|---|
| [`results/multistyle_routing_gap_v1_review_20260826.csv`](results/multistyle_routing_gap_v1_review_20260826.csv) | 24/24 Stage 1 and 24/24 true Stage 2 visual decisions: Comic 6/6 Stage 1, Ink 5/6 Stage 2, Watercolor 6/6 Stage 1, Needle-felt 0/6 | completed full-bank review |
| [`results/origami_v1_residual_stage2_review_20260826.md`](results/origami_v1_residual_stage2_review_20260826.md) | V1-100 residual Stage 2 rescues 0/3 strict hard failures; `011` is local face improvement only | completed final inference diagnostic |
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

The final returned evidence also remains outside Git:

| Returned folder | Comparison | Repository decision record |
|---|---|---|
| `multistyle-routing-gap-v1` | 24 Stage 1 outputs and their 24 true sequential Stage 2 outputs | `results/multistyle_routing_gap_v1_review_20260826.csv` |
| `origami-lora-heldout-base-ckpt100-200-300-seed42/checkpoint-100` | frozen V1 checkpoint 100 inputs for the final residual edit | `results/origami_lora_holdout_review_20260824.md` |
| `origami-v1ckpt100-residual-stage2-hard3-seed42` | `002/011/018` after V1 output to Base residual Stage 2 | `results/origami_v1_residual_stage2_review_20260826.md` |

## Historical execution documents

Files named `AUTODL_*`, older `PROMPT_*`, and the detailed pair-bank workflow are preserved to show
how completed experiments were run. Their commands and “next experiment” language describe
historical state. All prescribed routing-gap inference is complete; no LoRA training, new dataset
generation, formal evaluation, or additional AutoDL run is authorized.
