# Multistyle adaptation branch: final result and routing report

## Scope and conclusion

This report closes the multistyle adaptation branch using completed real-model runs and existing
visual review only. No new image generation, LoRA training, metric run, or AutoDL session was used
for this closure.

The practical result is a routed system, not one universal adapter. Original-BF16 FLUX Kontext is
the default editor. Painting-like styles and Needle-felt do not justify a style LoRA in the current
project. Geometry/material-entangled styles remain harder: Origami has one useful but limited
adapter, while Clay and 3D cartoon require explicit fallback or failure handling. Closed-teacher
images are private reconstruction candidates, not ground-truth natural appearances.

## Evidence by style

| Style | What actually ran | Observed result | Final route | Evidence limit |
|---|---|---|---|---|
| Comic | A fixed five-source BF16 FLUX pilot; formal calibration and the reduced post-unblinding replication | FLUX gave a strong natural-reconstruction signal. In the reduced replication it passed 6/8 comic sources; the historical four-style calibration result was 37/40 overall for FLUX versus 1--3/40 for each SDXL method. | Base FLUX prompt-only. Record and review isolated failures; no LoRA. | The 6/8 count comes from a completion-informed reduced replication, not the abandoned confirmatory 300-candidate design. |
| Ink | The same pilot, calibration, and reduced-replication path | FLUX passed 6/8 ink sources in the reduced replication and was visibly stronger than the SDXL prompt/Canny plateau. | Base FLUX prompt-only. Record and review isolated failures; no LoRA. | Same reduced-sample and post-unblinding limitation. |
| Watercolor | The same pilot, calibration, and reduced-replication path | FLUX passed 6/8 watercolor sources in the reduced replication and showed the same painting-like-style capability signal. | Base FLUX prompt-only. Record and review isolated failures; no LoRA. | Same reduced-sample and post-unblinding limitation. |
| Needle-felt | A five-source exploratory Stage 1/Stage 2 prompt comparison with Base FLUX | Stage 1 usually removed most visible fiber cues; the alternate Stage 2 edit added little. This did not establish a need for adaptation. | Base FLUX Stage 1; use a targeted second edit only if fiber remains. No LoRA. | Small qualitative pilot; no separate strict pass count or broad-source claim. |
| 3D cartoon | Five-source BF16 FLUX pilot; 27-source curated Stage 1 bank; eight-source true sequential Stage 2 probe; one strict closed-teacher target; eight-pair rank-16/200-step LoRA smoke evaluated on five pilot sources | Base FLUX often retained exaggerated eyes, CGI geometry, material, and lighting. Sequential Stage 2 produced 0/8 strict targets. The LoRA loaded and changed inference but generally became more conservative and closer to the 3D input; it was not a reliable improvement. | Base FLUX as the reproducible first attempt; then closed teacher when authorized and appropriate, otherwise record an explicit failure. Freeze the negative LoRA smoke. | The LoRA used only eight low-drift synthetic pairs; there is only one strict teacher pair. Do not generalize the negative result to all possible data or adapters. |
| Clay | Five-source material pilot; 19-source curated Stage 1 bank; 12-source true sequential Stage 2 probe; one strict closed-teacher target | Stage 1 usually retained clay. Stage 2 sometimes naturalized the face but left clay clothing (`012`, `021`) or invented a different person (`013`, `015`); it produced 0/12 strict full-frame targets. | Stage 1, then a true Stage 1-to-Stage 2 residual-material edit; use a closed teacher only as a fallback, otherwise record failure. No LoRA with the current pair bank. | One strict teacher pair is insufficient for training. The teacher route is a proposed fallback, not a measured general solution. |
| Origami | 24-source pair-bank review; 23 accepted strict teacher pairs; V1 rank-16/300-step training; fixed six-source Base/V1 holdout; 28 additional hard pairs and fresh 51-pair V2 rank-16/200-step training; a three-source V2 prompt-alignment diagnostic; fresh 51-pair caption-only V2.1 rank-16/200-step training | Base passed about 1/6. Every V1 checkpoint passed about 3/6; checkpoint 100 was selected because later checkpoints added drift without rescuing cases. V2 reached at most 2/6 under the shared prompt; the source-specific diagnostic rescued only `011`, for an effective result near 3/6. V2.1 also reached at most 2/6. Hard failures retained paper hair/headwear, garments, busts, or supports; stronger removal could change age, beard color, or expression. | Base or optional frozen V1 checkpoint 100, always described as limited. For residual regions, use a true second edit; a closed teacher is a final fallback. Stop V2/V2.1. | Six holdouts are a small fixed qualitative set. V1 is a partial adaptation result, not a solved style. Teacher targets are reconstructions rather than ground truth. |

The reduced replication also found FLUX at 4/8 for 3D cartoon versus 6/8 for each painting-like
style. That comparison supports the routing boundary, but its completion-informed selection and
missing repeat round prevent confirmatory interpretation.

## Final routing contract

| Input style | Default | If material/style remains | Terminal handling | Adapter status |
|---|---|---|---|---|
| Comic | Base FLUX prompt-only | review the isolated case | retain output only with an honest qualitative decision, otherwise failure | none |
| Ink | Base FLUX prompt-only | review the isolated case | retain output only with an honest qualitative decision, otherwise failure | none |
| Watercolor | Base FLUX prompt-only | review the isolated case | retain output only with an honest qualitative decision, otherwise failure | none |
| Needle-felt | Base FLUX Stage 1 | targeted residual-fiber second edit | explicit failure if fiber or subject drift remains | none |
| Origami | Base FLUX or optional V1 checkpoint 100 | true residual-region second edit | closed teacher when authorized, otherwise explicit failure | freeze V1-100; stop V2/V2.1 |
| Clay | Base FLUX Stage 1 | true Stage 1-to-Stage 2 residual-material edit | closed teacher when authorized, otherwise explicit failure | insufficient strict pairs |
| 3D cartoon | Base FLUX | closed teacher when authorized | explicit failure | freeze negative eight-pair smoke |

The second-edit and teacher columns define handling options, not guaranteed pass paths. Do not
silently accept a result that removes material by changing the depicted subject, crop, clothing,
support, or composition.

## Cross-experiment interpretation

The main boundary is not simply “2D works, 3D fails.” FLUX handles many surface-style changes, and
Needle-felt often yields to one edit. Failure becomes persistent when style is encoded jointly in
facial geometry, large hair/headwear masses, clothing, busts, pedestals, or scene-wide material.
The Origami runs further show that more pairs and shorter captions are not sufficient by themselves:
full-frame supervision coverage and instruction alignment both matter, while stronger removal can
trade against subject preservation.

This makes the negative runs useful. The 3D smoke shows that conservative synthetic pairs can teach
near-copy behavior. Origami V2/V2.1 show that increasing a small pair bank from 23 to 51 and changing
caption form did not move the six-source generalization boundary. The appropriate project ending is
therefore routing and explicit failure reporting, not another automatic training cycle.

## Optional future work requiring new authorization

No extension below is active. If the operator later authorizes one additional adaptation study,
Clay is the first candidate: run at most a 6--8-source closed-teacher feasibility pilot and continue
only if roughly five targets preserve the complete subject and composition while removing clay over
the full frame. Accumulate about 20 strict pairs before considering LoRA training. Otherwise stop.

## Evidence map

The repository reports and local-only visual artifacts supporting this summary are catalogued in
[`../EXPERIMENT_EVIDENCE_INDEX.md`](../EXPERIMENT_EVIDENCE_INDEX.md). Images, returned runs, model
weights, and training packages remain outside Git.
