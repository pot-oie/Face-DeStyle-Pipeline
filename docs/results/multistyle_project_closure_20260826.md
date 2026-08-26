# Final multistyle result and processing-routing report

> Finalized 2026-08-26 after the missing 24-source Stage 1/true-sequential-Stage 2 run, the
> three-source Origami V1 residual-Stage 2 diagnostic, and implementation of the human-reviewed
> manifest/record router. No experiment is left active by this report.

## Scope and conclusion

This report closes both the adaptation branch and the missing processing-route branch. The final
completion run added 24 Stage 1 outputs and 24 true sequential Stage 2 outputs for Comic, Ink,
Watercolor, and the replacement Needle-felt bank. It also tested three frozen Origami V1 outputs
through a true residual edit. No new LoRA was trained.

The result is not one universal adapter or an automatic quality selector. Original-BF16 FLUX
Kontext is the default editor, while a lightweight executable router turns explicit human decisions
into terminal routes or next-stage input subsets. Painting-like styles work well through Base FLUX.
Needle-felt, Clay, 3D cartoon, and hard Origami cases demonstrate that material entangled with full
subject geometry still requires a teacher fallback or an honest failure. Closed-teacher images are
private reconstruction candidates, not ground-truth natural appearances.

## Evidence by style

| Style | What actually ran | Observed result | Final route | Evidence limit |
|---|---|---|---|---|
| Comic | Earlier pilot/calibration/reduced replication plus six new sources through Stage 1 and true sequential Stage 2 | New review: Stage 1 passed 6/6; Stage 2 added no material gain. Earlier reduced replication was 6/8. | Base FLUX Stage 1, then accept after review. Do not run Stage 2 by default. | The new six-source set is a small qualitative extension, not a population estimate. |
| Ink | Earlier pilot/calibration/reduced replication plus six new sources through Stage 1 and true sequential Stage 2 | New review: Stage 2 passed 5/6. `met-12464` still retained large ink masses after both edits. Earlier reduced replication was 6/8. | Base Stage 1, then Stage 2 for residual ink; teacher when authorized or explicit failure if ink remains. | The six sources mix museum drawings and synthetic portraits; one difficult drawing remains unresolved. |
| Watercolor | Earlier pilot/calibration/reduced replication plus six new sources through Stage 1 and true sequential Stage 2 | New review: Stage 1 passed 6/6; Stage 2 added no material gain and sometimes only redrew facial detail. Earlier reduced replication was 6/8. | Base FLUX Stage 1, then accept after review. Do not run Stage 2 by default. | The new six-source set is small and visually selected. |
| Needle-felt | Earlier five-source non-sequential prompt comparison plus a replacement six-source bank through Stage 1 and true sequential Stage 2 | New strict review: 0/6 full-frame passes. Some faces became more realistic, but doll proportions, textile garments, busts, and supports remained. | Base Stage 1 is only a first attempt; Stage 2 is optional diagnostic handling. Teacher when authorized or explicit failure. No LoRA. | Six difficult material-v2 busts define the observed boundary; they do not prove all Needle-felt inputs fail. |
| 3D cartoon | Five-source BF16 FLUX pilot; 27-source curated Stage 1 bank; eight-source true sequential Stage 2 probe; one strict closed-teacher target; eight-pair rank-16/200-step LoRA smoke evaluated on five pilot sources | Base FLUX often retained exaggerated eyes, CGI geometry, material, and lighting. Sequential Stage 2 produced 0/8 strict targets. The LoRA loaded and changed inference but generally became more conservative and closer to the 3D input; it was not a reliable improvement. | Base FLUX as the reproducible first attempt; then closed teacher when authorized and appropriate, otherwise record an explicit failure. Freeze the negative LoRA smoke. | The LoRA used only eight low-drift synthetic pairs; there is only one strict teacher pair. Do not generalize the negative result to all possible data or adapters. |
| Clay | Five-source material pilot; 19-source curated Stage 1 bank; 12-source true sequential Stage 2 probe; one strict closed-teacher target | Stage 1 usually retained clay. Stage 2 sometimes naturalized the face but left clay clothing (`012`, `021`) or invented a different person (`013`, `015`); it produced 0/12 strict full-frame targets. | Stage 1, then a true Stage 1-to-Stage 2 residual-material edit; use a closed teacher only as a fallback, otherwise record failure. No LoRA with the current pair bank. | One strict teacher pair is insufficient for training. The teacher route is a proposed fallback, not a measured general solution. |
| Origami | 24-source pair-bank review; V1/V2/V2.1 training and fixed holdouts; prompt-alignment diagnostic; final V1-100-to-Stage-2 diagnostic on `002/011/018` | Base passed about 1/6 and V1-100 about 3/6. V2/V2.1 did not beat V1. The final true residual edit rescued 0/3: `011` naturalized the face and scalp but retained a dominant folded garment. | Base or optional frozen V1 checkpoint 100, always described as limited. Residual Stage 2 is review-triggered, followed by teacher or explicit failure. Stop V2/V2.1. | Six holdouts are a small fixed qualitative set. V1 is partial, and the residual edit does not improve the aggregate strict count. |

The reduced replication also found FLUX at 4/8 for 3D cartoon versus 6/8 for each painting-like
style. That comparison supports the routing boundary, but its completion-informed selection and
missing repeat round prevent confirmatory interpretation.

## Final routing contract

| Input style | Default | If material/style remains | Terminal handling | Adapter status |
|---|---|---|---|---|
| Comic | Base FLUX Stage 1 | no default second edit | accept Stage 1 after review; otherwise failure | none |
| Ink | Base FLUX Stage 1 | true Stage 2 for residual ink | accept Stage 2, authorized teacher, or explicit failure | none |
| Watercolor | Base FLUX Stage 1 | no default second edit | accept Stage 1 after review; otherwise failure | none |
| Needle-felt | Base FLUX Stage 1 as a first attempt | optional Stage 2 diagnostic | authorized teacher or explicit failure; current replacement bank is 0/6 | none |
| Origami | Base FLUX or optional V1 checkpoint 100 | review-triggered residual Stage 2 | authorized teacher or explicit failure; residual hard-three rescue is 0/3 | freeze V1-100; stop V2/V2.1 |
| Clay | Base FLUX Stage 1 | true Stage 1-to-Stage 2 residual-material edit | closed teacher when authorized, otherwise explicit failure | insufficient strict pairs |
| 3D cartoon | Base FLUX | closed teacher when authorized | explicit failure | freeze negative eight-pair smoke |

The second-edit and teacher columns define handling options, not guaranteed pass paths. Do not
silently accept a result that removes material by changing the depicted subject, crop, clothing,
support, or composition.

## Cross-experiment interpretation

The main boundary is not simply “2D works, 3D fails.” FLUX handles many surface-style changes.
Failure becomes persistent when style is encoded jointly in
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

The exact 24-source decisions are recorded in
[`multistyle_routing_gap_v1_review_20260826.csv`](multistyle_routing_gap_v1_review_20260826.csv), and
the final Origami diagnostic is recorded in
[`origami_v1_residual_stage2_review_20260826.md`](origami_v1_residual_stage2_review_20260826.md).
