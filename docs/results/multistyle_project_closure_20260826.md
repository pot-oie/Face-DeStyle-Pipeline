# Final multistyle result report

## Scope and final conclusion

The multistyle experiment branch is closed. The last inference run processed 137 non-Origami
sources with original-BF16 FLUX.1-Kontext-dev through both Stage 1 and a true sequential Stage 2.
Both stages completed 137/137 records and images with zero recorded generation failures. Stage 2
used the corresponding Stage 1 image for every source; source IDs and style labels matched exactly.

The visual review selected 71/137 usable outputs and recorded 66 explicit failures. This is a
practical qualitative routing result, not a population estimate or formal test. No new LoRA,
teacher output, source data, metric suite, or seed search was added.

The project now has a real style-dependent route rather than a universal adapter claim. Comic,
Ink, and Watercolor usually work with one Base FLUX edit. Clay sometimes benefits from a true
second edit. 3D cartoon and Needle-felt remain unreliable because geometry, garments, supports,
and scene-wide material remain stylized. Origami V1 checkpoint 100 remains only a limited optional
adapter at about 3/6 on its fixed qualitative holdout.

## Final 137-source validation

The review used a simple full-frame boundary: the selected result had to look like a plausible
photograph while retaining the depicted person's recoverable attributes, pose, composition, and
major scene content. Removing the medium by visibly changing sex, age, identity cues, a key object,
or the subject construction was counted as failure.

| Style | Sources | Stage 1 pass | Stage 2 pass | Stage 2 rescue | Selected terminal route | Main observation |
|---|---:|---:|---:|---:|---|---|
| Comic | 24 | 23 | 23 | 0 | 23 Stage 1; 1 failure | Stage 1 reliably removed drawing cues. Stage 2 added no material gain. One sketch changed a key object and scene semantics. |
| Ink | 24 | 21 | 21 | 0 | 21 Stage 1; 3 failures | Both stages usually became photographic. The three failures changed major person attributes; Stage 2 mostly sharpened rather than rescued. |
| Watercolor | 24 | 21 | 21 | 0 | 21 Stage 1; 3 failures | Stage 1 was sufficient when successful. Three historical portraits changed major person attributes. |
| 3D cartoon | 24 | 0 | 0 | 0 | 24 failures | CGI/cartoon geometry, material, eye shape, or lighting remained after both edits. |
| Clay | 24 | 0 | 5 | 5 | 5 Stage 2; 19 failures | Stage 2 naturalized five faces without an observed major subject/pose change, but most outputs remained clay or sculpture. |
| Needle-felt | 17 | 1 | 1 | 0 | 1 Stage 1; 16 failures | One synthetic portrait became plausibly photographic; the rest retained felt, doll, bust, textile, or support construction. |
| **Total** | **137** | **66** | **71** | **5** | **71 successes; 66 failures** | Stage 2's only strict incremental gain was the five-source Clay subset. |

The exact decisions and notes are in
[`multistyle_routing_validation_137_review_20260827.csv`](multistyle_routing_validation_137_review_20260827.csv).
The five Clay rescues were `matv2-clay-003`, `012`, `015`, `020`, and `021`. The sole accepted
Needle-felt result was `synthetic-needle-felt-006`. No accepted Stage 1 result regressed to failure
under the binary review, but this does not imply Stage 2 improved it.

## Actual processing route

The machine-readable policy is [`../../configs/multistyle_routing.yaml`](../../configs/multistyle_routing.yaml).
All routes remain review-gated because no automatic quality selector was trained.

| Input style | Default route | If the result remains unacceptable | Adapter status |
|---|---|---|---|
| Comic | Base FLUX Stage 1 | explicit failure | none |
| Ink | Base FLUX Stage 1 | Stage 2 only when visible ink remains; otherwise explicit failure | none |
| Watercolor | Base FLUX Stage 1 | explicit failure | none |
| 3D cartoon | Base FLUX Stage 1 as diagnostic only | separately authorized external teacher, otherwise explicit failure | negative eight-pair LoRA smoke frozen |
| Clay | Base FLUX Stage 1 followed by true sequential Stage 2 | separately authorized external teacher, otherwise explicit failure | current strict pair bank is insufficient; no LoRA |
| Needle-felt | Base FLUX Stage 1 as diagnostic only | separately authorized external teacher, otherwise explicit failure | none |
| Origami | Base FLUX Stage 1 | optional frozen V1 checkpoint 100, then explicit failure if still unacceptable | V1-100 retained as limited; V2/V2.1 stopped |

An external teacher is an optional future fallback, not a measured success path and not active
work. The router must never silently accept a material-removal result that changed the subject or
pretend that an unavailable teacher output exists.

## Relation to earlier experiments

The larger run confirms the earlier boundary while refining Clay:

- the earlier six-source completion batch found Comic 6/6 at Stage 1, Ink 5/6 at Stage 2,
  Watercolor 6/6 at Stage 1, and Needle-felt 0/6;
- the earlier true sequential probes found 3D cartoon 0/8 and Clay 0/12 strict full-frame targets;
- the new 24-source Clay bank shows a small 5/24 Stage 2 rescue signal, but not enough reliability or
  paired supervision to justify a LoRA;
- Origami Base was about 1/6 and frozen V1-100 about 3/6; V2/V2.1 and the final hard-three residual
  edit did not improve the aggregate boundary.

These are qualitative, source-specific counts. Closed-teacher targets are reconstruction
candidates, not ground-truth natural appearances.

## Evidence and storage boundary

The returned run remains outside Git at:

```text
/Users/pot/Desktop/multistyle-routing-validation-137-v1
```

The local comparison pages remain outside Git at:

```text
/private/tmp/multistyle-validation-137-review
```

Git contains only the compact manifest, decisions, routing policy, reports, and tests. It does not
contain source images, the 274 generated images, model weights, checkpoints, caches, or bulk logs.
The concise evidence map is [`../EXPERIMENT_EVIDENCE_INDEX.md`](../EXPERIMENT_EVIDENCE_INDEX.md).

No experiment is left active. Further training, teacher generation, new data collection, or
additional large-batch inference requires a new operator decision rather than being treated as the
next step of this branch.
