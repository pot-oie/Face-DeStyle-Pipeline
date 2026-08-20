# Agent instructions for Face-DeStyle-Pipeline

This file is the required entry point for any agent working in this repository. Read it before
changing code, configs, data, or documentation. Then read `docs/research_context.md`,
`docs/data_acquisition.md`, `docs/HANDOFF_AUTODL.md`, and the task-specific files and tests.

## Execution environments

This repository is developed across two distinct environments. Do not assume that the current
machine has the capabilities of the other environment.

### Local macOS workspace

- Used only for lightweight development, documentation, mocks, Ruff, and pytest.
- The Conda environment is named `face-destyle` and uses Python 3.10.
- Do not install Torch, Diffusers, Transformers, CUDA, ONNX Runtime GPU, InsightFace, or other heavy
  model runtimes locally unless the user explicitly changes this policy.
- Never download model weights during local tests. Heavy functionality must use injected mocks,
  lazy imports, or clear `NotImplementedError`/runtime guidance.

### AutoDL server

- The persistent project root is normally `/root/autodl-tmp/face-destyle`, supplied through
  `FACE_DESTYLE_ROOT`. Source code is normally below `code/Face-DeStyle-Pipeline`.
- AutoDL may be started in no-GPU mode. Always check `nvidia-smi` and `torch.cuda.is_available()`
  before GPU work; a prepared CUDA environment does not prove that a GPU is currently allocated.
- Model weights, datasets, caches, checkpoints, and bulk outputs stay on the AutoDL data disk and
  must not enter Git.
- Resolve server model paths through `configs/models.yaml`, `FACE_DESTYLE_ROOT`, `HF_HOME`, and
  `HF_HUB_CACHE`; never hard-code the server path in Python source.
- Run `python scripts/check_model_assets.py` before assuming a downloaded model is present.
- Download completion/file integrity is not GPU-load, inference, metric, or scientific validation.

## AutoDL networking

Mainland access to GitHub and Hugging Face may be slow or unstable. Before downloading anything,
prefer the provider's current official instructions and the repository documentation. Do not
silently switch mirrors or download a second copy into another cache.

- Persistent Hugging Face cache must be on the data disk:
  `HF_HOME=$FACE_DESTYLE_ROOT/cache/huggingface` and `HF_HUB_CACHE=$HF_HOME/hub`.
- AutoDL's documented academic proxy is enabled in an interactive shell with
  `source /etc/network_turbo`. It is not guaranteed to be stable and should be unset when it harms
  normal network access.
- The public Hugging Face mirror uses `HF_ENDPOINT=https://hf-mirror.com`. Do not combine it with
  stale proxy variables unless deliberately testing that route.
- For large Hugging Face files, the server has used `hfd.sh` with `aria2c`; rerunning the same
  command resumes. Audit `.aria2` and `.incomplete` files before deleting them.
- Qwen weights were obtained from ModelScope because direct Hugging Face Xet downloads were very
  slow. Local ModelScope directories are valid model sources and need not be duplicated into the
  Hugging Face cache.
- Authentication tokens, proxy credentials, SSH endpoints, cookies, and signed URLs are secrets or
  ephemeral state. Never write them into tracked files, logs, tests, or handoff templates.

## Required safety and reproducibility behavior

- Check the working tree before editing and preserve user changes.
- Pin upstream revisions when available and record local-file checksums for ModelScope or standalone
  checkpoints.
- Review third-party weight licenses independently of this repository's Apache-2.0 license.
- Do not report copy-backend, smoke-test, mock, downloaded-file, or uncalibrated metric output as a
  research result.
- Run Ruff, pytest, and every new script's `--help` locally before handoff.
- Update `docs/HANDOFF_AUTODL.md` when environment state, model inventory, or next-server commands
  materially change.
- After a server experiment, give the user a `scripts/package_run.py` command that writes and
  verifies a ZIP plus SHA-256 before `--cleanup` removes only that selected run directory. Never
  recommend broad wildcard deletion of outputs, models, caches, or datasets.

## Mission and claim boundary

Build a compact, reproducible face-domain destylization study that can be completed on one AutoDL
GPU. The study asks whether style-adaptive prompts and structural conditions improve the trade-off
between removing artistic appearance and preserving subject, scene, recoverable facial identity,
and pose. A negative or mixed result is acceptable; training a state-of-the-art model is not the
goal.

This is an independent undergraduate reproduction in the same research direction as *Learning to
Stylize by Learning to Destylize*. It is not the official DeStyle implementation and does not own
or reproduce DeStyle-350K, DeStylePipe, DestyleCoT-Filter, or BCS-Bench at published scale. Do not
copy paper claims into results. A destylized face is a model reconstruction, not the person's true
or ground-truth appearance.

## Active milestone

The 20-source SDXL Base pilot now covers generic and adaptive prompt-only generation, global Canny,
and region-aware Canny. It did not find a stable adaptive-prompt or Region Canny advantage, and
further SDXL Base prompt, strength, or Canny-weight scanning is stopped. This is a plateau for the
tested SDXL Base configuration, not a claim about the theoretical limit of the SDXL family.

The original-BF16 `FLUX.1-Kontext-dev` generator-capability probe is complete on the frozen
20-source pilot. It used native 1024x1024 source-image-plus-instruction editing, CPU/model offload,
and no structural condition. All 20 records succeeded in one pipeline-loading session. Unblinded
visual review found a capability signal for comic, ink, and watercolor sources but persistent
rendered geometry and materials on all five 3D-cartoon sources. This review is exploratory and
uncalibrated, not formal evaluation or a strict cross-model comparison: the SDXL pilot used 768x768,
and equal seeds do not imply matched noise across architectures. Freeze these FLUX outputs. Do not
scan parameters or add FLUX Canny, Depth, Pose, LoRA, or quantization.

Primary raw metric execution is complete for the four SDXL methods (80 pairs) and FLUX (20 pairs):
DINO, CLIP, and Qwen completed 100/100 records; ArcFace produced 99 cosine scores plus one explicit
SDXL no-face result. Global Canny leads source-similarity metrics, consistent with contour retention,
while Qwen gives all five methods the same mean style-removal score and does not reproduce the visual
FLUX capability signal. Do not rank or accept methods from these uncalibrated pilot metrics.

The source inventory is now frozen in `data/manifests/formal-v1/inputs.jsonl`: 20 pilot/debug,
40 calibration, and 60 held-out test sources, balanced 5/10/15 per style. At freeze time all files
matched their SHA-256 values and `source_id`, `source_group_id`, and file SHA-256 had zero overlap
between splits. The manifest is private-research runnable metadata, not an image redistribution
package. The test split remains sealed until blinded human calibration and all thresholds/routing
rules are frozen. The active task is calibration blind review and threshold/routing design; formal
generation and calibration source-similarity metric execution are complete.

A FLUX seed-stability extension completed 20/20 outputs for both seed 43 and seed 44. The complete
seed-44 archive passed CRC, record, and image validation with SHA-256
`9c2042df4b4ee81074467b0cc2f395d044421f719e49a5cb3d84140d5fcce591`. Stop adding seeds. Keep
seeds 43/44 separate from the primary seed-42 evaluation and do not select a favorable seed or use
the extension for parameter tuning.

Formal generation is complete: each of the four frozen SDXL methods has 100/100 new
calibration+test records with zero failures, and FLUX has 40/40 calibration records with zero
failures. All five returned archives and SHA-256 sidecars passed checksum, ZIP CRC, manifest-record,
output-file, decode, mode, and resolution checks locally. Do not generate more formal candidates
now. Preserve test outputs unopened. Calibration-only DINO, CLIP, and paired ArcFace execution is
complete for 200/200 method-source pairs with no evaluator failures; ArcFace yielded 197 cosine
values and three explicit generated-output no-face statuses. One complete 200-pair method-hidden
human round is now frozen and unblinded; FLUX passed 37/40 while each SDXL method passed 1--3/40
under the all-dimensions-at-least-4 rule. Round B was waived before unblinding, so repeat agreement
is unavailable. Failure-type blanks mean unreported, not no failure. The frozen test rule uses FLUX
first and rejects its failures; existing SDXL fallbacks rescued 0/3 FLUX calibration failures.

The operator has adopted a narrow private-research interpretation under which ArcFace/InsightFace
may be used only as a paired face-drift diagnostic for these fixed experimental inputs and outputs.
Do not turn it into identity search, identification, authentication, surveillance, enrollment, or a
real-person recognition system; do not publish reusable face templates. Record no-face failures and
the operator interpretation with any result. This operational boundary is not legal advice or a
claim that the license text is unambiguous.

Primary evaluation uses DINOv2 Base, CLIP ViT-L/14, InsightFace where a face is detectable, a
structured Qwen2.5-VL-3B style-removal rubric, and blinded human review. Depth, Refiner, InstantID,
RealVisXL, Florence, 7B VLM auditing, and LoRA training remain secondary.

## Research discipline

- Hold input image, seed, resolution, scheduler, steps, guidance, and strength constant when
  comparing one experimental factor.
- Preserve raw outputs. Never overwrite a prior run; use deterministic run IDs and separate
  directories.
- Record failures as data: OOM, no face, safety rejection, black output, changed subject, structure
  drift, residual style, and corrupted input.
- Calibrate thresholds on a human-labeled calibration split and freeze them before opening test.
- Report distributions, paired differences, sample counts, failures, and uncertainty rather than
  selecting only visually attractive examples.
- Keep pilot, calibration, and test sources disjoint by `source_id`; near duplicates share one
  source group.
- Do not change the primary matrix after seeing test results without labeling the change as a new
  exploratory experiment.

## Data rules

- Follow `docs/data_acquisition.md` for acquisition, licensing, QC, and split assignment.
- Do not crawl Pinterest, social media, portfolio sites, search-result thumbnails, or unlicensed
  WikiArt mirrors.
- Do not commit raw faces, bulk artworks, caches, embeddings, or model weights.
- Maintain provenance keyed by `source_id`, including landing page, provider/object ID, rights
  statement, acquisition date, checksum, and QC decision.
- Existing archives are pilot material until provenance is reconstructed; otherwise label them
  `legacy_private` and exclude them from public release and license-sensitive claims.
- Prefer CC0/public-domain artworks or explicitly licensed synthetic identities over private-person
  photographs. Never use outputs for identification or true-appearance claims.

## Triplet semantics

For an accepted source artwork `A`, `destylized_content_path` is the generated reconstruction,
`original_style_target_path` is unchanged artwork `A`, and `style_reference_path` is a different
source `B` in the same style category. The current proxy guarantees a different `source_id`; do not
call it cross-semantic matching unless the schema and sampler enforce distinct semantic categories.

## Definition of a reportable result

A result is reportable only when inputs have recorded provenance and a frozen split; the real model
ran on the stated GPU; records contain the exact model revision and settings; formal metrics rather
than smoke sentinels were computed; thresholds were calibrated without test leakage; outputs were
manually audited under the written rubric; comparisons used matched sources and settings; and any
summary includes limitations and failure counts.
