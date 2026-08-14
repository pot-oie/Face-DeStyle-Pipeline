# Research context and execution contract

## 1. Why this project exists

Supervised style transfer needs three pieces of information:

1. **content** -- the subject, pose, geometry, composition, and scene to preserve;
2. **style condition** -- the appearance to transfer;
3. **target** -- the desired stylized output used for supervision.

A forward synthetic-data pipeline starts from a natural content image and asks an existing model to
create the target artwork. This is convenient, but artifacts, weak style fidelity, and content
changes become errors in the target itself. A model trained on those targets inherits the ceiling
of the model that produced them.

Destylization reverses which side is generated. It starts from an existing high-quality stylized
image, keeps that image unchanged as the target, and generates a natural-looking reconstruction to
serve as content. This moves generation errors to the input side rather than the target side. It
does not make the pair perfect: the reconstruction can change identity, geometry, objects, text,
lighting, or composition. The research problem is therefore not merely generation; it is the
joint problem of **destylization, structural control, and quality filtering**.

## 2. What this reproduction is trying to establish

The project is deliberately narrower than the reference paper. It studies stylized portraits and
asks three research questions:

### RQ1 -- Prompt adaptation

Does a style-category prompt remove more residual style than one generic instruction without
reducing content or identity preservation?

Expected comparison: `prompt_generic` vs. `prompt_adaptive` with the same input, seed, model,
resolution, scheduler, steps, guidance, and image-to-image strength.

### RQ2 -- Structural conditioning

Does adding Canny structure improve content preservation, and does separating face and background
conditions offer a better trade-off than one global Canny map?

Expected comparison: `prompt_adaptive` vs. `global_canny` vs. `region_canny`. Canny may preserve
stylized contours as well as desired geometry, so an improvement is a hypothesis, not an assumed
fact.

### RQ3 -- Failure-aware progressive routing

After uniform prompt-only and structural methods reach a plateau, does routing samples by a
predeclared failure reason to the next eligible method improve accepted-pair yield relative to
applying one method to every sample? Routing rules must be fixed before observing the comparison
outputs. Pose remains a possible later extension for an explicitly justified failure class, not the
current research question.

### Exploratory generator-capability probe

The SDXL Base pilot reached a practical plateau: prompt adaptation was not consistently better,
global Canny retained artistic contours, and region-aware Canny did not show a stable improvement.
This does not establish an SDXL-family limit. Before adding more structural controls, compare the
existing SDXL Base adaptive prompt-only output with native image editing from original BF16
`FLUX.1-Kontext-dev` on the same preselected sources. The probe asks whether generator capability
and image-editing pretraining improve natural reconstruction without unacceptable content drift.

This is an exploratory generator extension, not a strict DeStyle reproduction or a new primary RQ.
Cross-model seeds do not represent matched noise. Hold source image, task semantics, and resolution
constant while keeping each generator's configuration fixed and fully recorded. Start with one
source per primary style; four examples can establish engineering feasibility or a capability signal
but cannot support a reportable scientific conclusion.

## 3. Hypotheses and acceptable outcomes

- **H1:** adaptive prompts increase human/VLM style-removal scores relative to a generic prompt.
- **H2:** global Canny increases structural similarity but can reduce style removal by preserving
  artistic contours.
- **H3:** region-aware Canny improves the content/style Pareto trade-off on faces compared with
  global Canny.
- **H4:** a predeclared failure-aware route improves accepted-pair yield over a uniform method
  without hiding failures or relaxing frozen quality criteria.

Mixed or negative findings are valid. Do not tune or hide samples until all hypotheses appear true.
The project succeeds if it produces a controlled and interpretable comparison.

## 4. Terminology

- **Source artwork / original style target:** unchanged stylized input.
- **Destylized content:** generated natural-looking reconstruction. It is not ground truth.
- **Style category:** coarse routing label such as `comic`, `ink`, or `watercolor`.
- **Generic prompt:** one instruction shared by all categories.
- **Adaptive prompt:** category-specific instruction describing what visual cues to remove.
- **Structural control:** Canny, region-aware Canny, and optionally pose.
- **Accepted pair:** a candidate passing calibrated content and style-removal gates and manual
  policy checks.
- **Triplet:** destylized content + different-source style reference + unchanged source target.
- **Identity preservation:** similarity of recoverable facial evidence, not proof of a person's
  real identity or true appearance.

The style label is metadata used for routing, balancing, and sampling. A text prompt is an
instruction condition. Neither is the pixel target; the unchanged artwork is the target.

## 5. Relationship to the reference paper

The reference paper scales the reverse-supervision idea to broad art and content domains with a
three-stage pipeline, VLM reasoning filters, multiple reference matches, a large dataset, and
downstream model training. This repository only claims an independent face-domain reproduction of
selected ideas.

Important differences:

| Topic | Reference paper | This repository |
|---|---|---|
| Domain | Broad art and 100+ content classes | Stylized faces/portraits |
| Source scale | 110K style images | Small authorized study set |
| Pipeline | Three progressive stages | Prompt baseline, adaptive prompt, structural controls |
| Filtering | Published CoT-style VLM filter | Metrics + structured VLM rubric + human calibration |
| Triplet reference | Cross-semantic category | Same style, different source/identity proxy |
| Downstream training | Multiple large backbones | Out of scope until the primary study is complete |

Never use the paper's dataset size, model performance, benchmark, authorship, or method names as
this project's result.

## 6. Minimal dataset and split

### Recommended first formal set

Start with four categories that have enough authorized data and visually distinct cues:

- `comic`;
- `3d_cartoon`;
- `ink`;
- `watercolor`.

Target **30 usable sources per category** (120 total), not thousands. Use source-group splits:

- pilot/debug: 5 per category;
- calibration: 10 per category;
- held-out test: 15 per category.

If the first four categories are stable, add `cyberpunk` and generic `animation` as extensions.
Do not add styles merely to raise a count. Each category needs an operational visual definition and
enough source diversity.

### Source independence

Near-duplicate crops, re-encodes, alternate generations from the same prompt/seed/source, and
paired halves from one composite image share one `source_group_id` and must stay in one split.
Use perceptual hashes as a screening aid, followed by visual verification.

## 7. What counts as a suitable source image

Prefer images with:

- a visible face large enough for identity/landmark evaluation;
- one primary subject during the first pilot;
- resolvable pose and scene structure;
- no watermark, UI overlay, frame, caption, or collage boundary;
- at least 768 pixels on the short side when possible;
- a style that is genuinely non-photographic and belongs to the declared category;
- provenance and rights sufficient for the intended private evaluation or public example use.

Exclude from the primary study:

- photographs that are merely color graded;
- extremely abstract faces with insufficient recoverable structure;
- heavy occlusion, tiny faces, or multi-panel collages;
- illegal, unsafe, sexual, or exploitative content;
- public figures or private persons where use would create likeness/privacy risk;
- uncertain licenses, missing landing pages, or images copied from search results/social platforms.

Keep hard cases in a separately labeled extension set; do not silently remove them after observing
method failures.

## 8. Pipeline semantics

### Stage A -- metadata and provenance

Acquire images under `docs/data_acquisition.md`, compute stable IDs/checksums, assign style and split,
and validate file integrity. No model inference should start on an untracked directory.

### Stage B -- prompt-only baselines

Run generic and adaptive prompts on the same pilot sources. The current SDXL img2img backend is the
first executable baseline. Inspect outputs before any batch expansion and record VRAM/runtime.

### Stage C -- structural ablations

Add global Canny, then production face parsing and region-aware Canny. Visually validate masks and
control maps. Pose is only a later extension if an observed, predeclared failure class justifies it.
A central mask is for smoke testing and cannot be reported as face segmentation.

### Stage D -- evaluation and filtering

Compute formal metrics and structured VLM judgments. Calibrate acceptance thresholds on the
calibration split against blinded human labels. Freeze the configuration, evaluate the test split
once, then manually audit every accepted test output.

### Stage E -- triplet construction

For an accepted target, choose a reference with the same style category and a different source ID.
In this face-domain study that mainly enforces different identity/source. The current schema cannot
guarantee paper-style cross-semantic selection; do not imply otherwise.

## 9. Human annotation rubric

Use a 0--5 ordinal scale for each dimension, with source and output displayed at the same size and
method identity hidden.

### Content preservation

- 5: subject, pose, composition, major objects, and spatial relations are preserved;
- 4: minor local drift, but all important structure remains;
- 3: recognizable content with a meaningful structural change;
- 2: multiple major changes;
- 1: weak correspondence;
- 0: unrelated or failed output.

### Style removal

- 5: output reads as a natural photograph with no meaningful residual target style;
- 4: small residual cues but predominantly photographic;
- 3: mixed photographic and artistic appearance;
- 2: strong residual style;
- 1: almost unchanged style;
- 0: failed/corrupt output.

### Recoverable facial identity

- 5: all visible identity-bearing geometry remains consistent;
- 4: minor changes, clearly the same depicted subject within what the artwork supports;
- 3: ambiguous but related;
- 2: substantial identity drift;
- 1: different face;
- 0: no usable face.

Map rubric scores to `[0, 1]` only through a documented transformation. An initial human acceptance
rule may require content >= 4 and style removal >= 4, with identity >= 4 when a valid face judgment
is possible. Final thresholds must be calibrated; the YAML defaults are placeholders.

## 10. Quantitative evaluation

- DINOv2: global/structural content similarity; can be influenced by style.
- CLIP: semantic consistency; not a substitute for geometric alignment.
- ArcFace: face embedding similarity when both images yield valid detections; log no-face cases
  rather than silently dropping them.
- Qwen2.5-VL-3B: structured style-removal and content rubric; output machine-parseable JSON with an
  explanation, but do not expose hidden chain-of-thought or treat the model as ground truth.
- Human review: calibration authority and final audit.

Report per-style distributions and paired method differences. A single aggregate average is not
enough. The smoke pixel similarity and sentinel style score are never formal metrics.

## 11. Primary success criteria

The first reportable milestone is complete when:

1. at least four styles have frozen source-group-disjoint pilot/calibration/test splits;
2. all four primary methods run on the same test sources with recorded settings;
3. formal DINO, CLIP, ArcFace failure handling, and Qwen rubric execution are validated;
4. thresholds are calibrated only on calibration data;
5. the test split is evaluated once and all accepted outputs are manually audited;
6. the report includes sample counts, pass rates, per-style metrics, paired differences, failure
   taxonomy, qualitative examples, compute/runtime, licenses, and limitations;
7. code tests pass and the run can be repeated from metadata/config without manual file renaming.

## 12. Explicitly out of scope until then

- tens of thousands of images;
- claiming a new algorithm or state of the art;
- full DeStyle-350K or BCS-Bench reproduction;
- training a general style-transfer backbone or LoRA;
- treating VLM scores as objective truth;
- adding every downloaded model to one ensemble;
- using test results to tune prompts or thresholds;
- building a web UI before the experiment is stable.

## 13. Sub-agent task protocol

Every delegated task should state:

- the active research question it advances;
- exact inputs and outputs;
- files allowed to change;
- acceptance tests;
- whether GPU execution is required;
- what must not be claimed from the result.

Agents should finish one vertical slice at a time. Example: implement global Canny ControlNet with
mocked tests, run one authorized pilot image on GPU, record metadata and failure behavior, then hand
off. Do not simultaneously add face parsing, pose, metrics, and reporting in one unreviewable patch.
