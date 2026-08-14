# Agent operating context

This file is the required entry point for any agent working in this repository. Read it before
changing code, configs, data, or documentation. Then read:

1. `docs/research_context.md` -- the research question, hypotheses, terminology, claim boundary,
   active milestone, and definition of done;
2. `docs/data_acquisition.md` -- how to find, license, screen, record, and split artistic images;
3. `docs/HANDOFF_AUTODL.md` -- current model inventory and the next GPU execution order;
4. the task-specific implementation file and its tests.

## Mission

Build a compact, reproducible **face-domain destylization study** that can be completed on one
AutoDL GPU. The study asks whether style-adaptive prompts and structural conditions improve the
trade-off between:

- removing artistic appearance;
- preserving subject/scene content;
- preserving recoverable facial identity and pose.

The intended output is an evidence-backed undergraduate reproduction: executable code, a small
authorized dataset, frozen experiment declarations, calibrated evaluation, ablations, qualitative
failure cases, and a factual report. A negative or mixed result is acceptable. The project is not
required to train a state-of-the-art style-transfer model.

## Relationship to the reference paper

This is an independent, small face-domain study in the same research direction as *Learning to
Stylize by Learning to Destylize*. It is **not** the official DeStyle implementation and does not
own or reproduce DeStyle-350K, DeStylePipe, DestyleCoT-Filter, or BCS-Bench at their published
scale.

The paper is conceptual context. Do not copy paper claims into results. In particular:

- a downloaded model is not a verified result;
- an experiment declaration is not a completed experiment;
- the copy backend and smoke metric are software tests, never scientific evidence;
- this repository's triplet builder currently guarantees same style and different `source_id`,
  not the paper's cross-semantic-category reference selection;
- face-only samples cannot demonstrate general arbitrary style transfer;
- a destylized face is a model reconstruction, not the person's true or ground-truth appearance.

## Active milestone

Complete the smallest defensible primary matrix before implementing extensions:

1. `prompt_generic`;
2. `prompt_adaptive`;
3. `global_canny`;
4. `region_canny`;
5. `canny_plus_pose` only after the first four are stable.

Primary evaluation uses DINOv2 Base, CLIP ViT-L/14, InsightFace where a face is detectable, a
structured Qwen2.5-VL-3B style-removal rubric, and blinded human review. Robustness models and
generation extensions are secondary. Do not begin Depth, Refiner, InstantID, RealVisXL, Florence,
7B VLM auditing, or LoRA training merely because weights are available.

## Research discipline

- Hold input image, seed, resolution, scheduler, steps, guidance, and strength constant when
  comparing one experimental factor.
- Preserve raw outputs. Never overwrite a prior run; use deterministic run IDs and separate
  directories.
- Record failures as data: OOM, no face, safety rejection, black output, changed subject, structure
  drift, residual style, and corrupted input.
- Calibrate thresholds on a human-labeled calibration split. Freeze them before opening the test
  split.
- Report distributions, paired differences, sample counts, failures, and uncertainty. Do not
  select only visually attractive examples.
- Keep pilot, calibration, and test sources disjoint by `source_id`. Near duplicates belong to the
  same source group.
- Do not change the primary matrix after seeing test results without labeling the change as a new
  exploratory experiment.

## Data rules

- Follow `docs/data_acquisition.md` exactly.
- Do not crawl Pinterest, social media, portfolio sites, search-result thumbnails, or unlicensed
  WikiArt mirrors.
- Do not commit raw faces, bulk artworks, caches, embeddings, or model weights.
- For every formal source image, maintain a provenance sidecar keyed by `source_id`, including the
  landing page, provider/object ID, rights statement, acquisition date, checksum, and QC decision.
- Existing team archives are useful pilot material but are not automatically formal evaluation
  data. Reconstruct provenance first; otherwise label them `legacy_private` and exclude them from
  public release and license-sensitive claims.
- Avoid real private-person photographs. Prefer CC0/public-domain artworks or explicitly licensed
  synthetic identities. Never use outputs for identification or claims about true appearance.

## Triplet semantics

For an accepted source artwork `A`:

- `destylized_content_path`: generated natural-looking reconstruction of `A`;
- `original_style_target_path`: unchanged source artwork `A`, which supplies the target appearance;
- `style_reference_path`: a different source `B` from the same style category.

The style label routes prompts, balancing, and evaluation; it is not the pixel-level target. The
original artwork is the target. The current face-domain proxy uses a different identity/source as
the reference. Do not call that cross-semantic matching unless the schema and sampler are extended
to record and enforce distinct semantic categories.

## Implementation expectations

- Prefer small, tested changes that advance the active milestone.
- Update schemas/configs/docs/tests together when a record format changes.
- Inject or mock heavyweight pipelines in unit tests; tests must not download models.
- Default real loaders to pinned revisions and `local_files_only: true`.
- Keep device paths in CLI/config/environment variables, never hard-code one AutoDL host path into
  library code.
- Before handing off, run `ruff check .`, `pytest`, and `git diff --check`; state which GPU commands
  were actually run and which remain unverified.

## Definition of a reportable result

A result is reportable only when all of the following are true:

- inputs have recorded provenance and a frozen split;
- the real model loaded and generated outputs on the stated GPU;
- run metadata records exact model revision and inference settings;
- formal metrics, not smoke sentinels, were computed;
- thresholds were calibrated without test leakage;
- outputs received a manual audit under the written rubric;
- comparisons use the same source images and controlled settings;
- limitations and failure counts accompany any summary number.
