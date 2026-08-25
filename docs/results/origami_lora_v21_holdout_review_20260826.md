# Origami LoRA V2.1 CLIP-safe holdout review (2026-08-26)

## Outcome

The caption-only V2.1 intervention did not improve Origami generalization. It copied the same 51
condition/target pairs as V2, replaced the 40 long instructions with five concise CLIP-safe
templates distributed `11/10/10/10/10`, and trained a fresh Base-model rank-16 adapter for 200
steps at learning rate `1e-4` and effective batch 4.

The training run is complete. Checkpoints 50, 100, 150, and 200 plus the final adapter are
structurally valid 108,435,512-byte safetensors files with matching tensor keys, dtypes, and shapes.
The log reached `200/200`. The later disconnect during three-shard Base-model loading occurred
after weight saving and did not truncate the LoRA.

All four checkpoints were evaluated on the same six holdouts with the shared CLIP-safe core prompt,
seed 42, 28 steps, guidance 2.5, and LoRA scale 1.0. The conservative strict result is:

| Checkpoint | Strict passes | Interpretation |
|---|---:|---|
| 50 | 0/6 | Paper hair, garments, or supports remain throughout. |
| 100 | 2/6 | `007` and `023` pass; the three hard regional failures remain. |
| 150 | 2/6 | Same useful boundary as checkpoint 100. |
| 200 | 2/6 | Same boundary; no late rescue. |

## Per-source result

| Holdout | Result | Rationale |
|---|---|---|
| `002` | fail at all checkpoints | Skin becomes somewhat smoother, but hair, layered clothing, bust, pedestal, and support retain paper geometry. |
| `007` | pass from 100 | Face, short hair, and clothing become sufficiently natural while broadly preserving pose and identity evidence. |
| `011` | fail at all checkpoints | Every checkpoint invents dark hair on the bald subject, and large garment regions retain constructed paper planes. |
| `018` | fail at all checkpoints | The central face becomes natural, but the large outer hair/headwear mass and shoulder garment remain folded paper. |
| `023` | pass from 100 | Face, hood, and garment are natural enough with pose, gaze, accessory, and palette retained. |
| `030` | fail at all checkpoints | Paper is removed strongly, but gray facial hair becomes black, apparent age decreases, and the expression becomes a broad smile. |

## Decision

Do not promote a V2.1 checkpoint, continue past 200 steps, increase rank, or perform another prompt
sweep. V1 checkpoint 100 remains the frozen selected Origami adapter at 3/6. The combined V2 and
V2.1 evidence shows that caption length and train/inference wording were secondary contributors,
not the primary cause of the full-subject failure. Any further single-pass LoRA experiment must
change the visual supervision with exact-category full-frame pairs; otherwise the next practical
direction is a true second residual-region edit rather than another caption-only retrain.

## Artifacts

- returned loose run:
  `/Users/pot/Documents/大创/实验归档/returned-runs/origami-lora-v21-heldout-checkpoints50-100-150-200-clip77-seed42`
- five-column review sheet:
  `/Users/pot/Documents/大创/实验归档/origami-lora-v21-review-20260826/overview.jpg`
- checkpoint 200 SHA-256:
  `32e762410b592f48760d86f78568989541999babb242439551565c3918f51c87`
- final adapter SHA-256:
  `d6cb7b3a59ceb5d3afbb505bdeeee8e68fcbbd0d350393733402109393be5e17`
