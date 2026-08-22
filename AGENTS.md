# Agent instructions for Face-DeStyle-Pipeline

This file is the required entry point for any agent working in this repository. Read it before
changing code, configs, data, or documentation. Then read `docs/research_context.md`,
`docs/data_acquisition.md`, `docs/HANDOFF_AUTODL.md`, and the task-specific files and tests.

## Operator policy: do not over-process experiments

This project now prioritizes useful experimentation and a clear research narrative over formal
evaluation ceremony. This applies to exploratory runs **and** to work previously described as a
formal, frozen, blinded, held-out, or confirmatory evaluation. Unless the operator explicitly asks
for one in a future task, do not require or propose blind testing, visual sealing, acceptance gates,
freeze markers, preregistration, repeat-rating rounds, formal archive validation, cryptographic
hashes, checksum sidecars, ZIP CRC checks, elaborate preflight blocks, complex log audits, or formal
statistical testing.

Older documents describing those procedures are historical records, not active execution contracts.
They may explain how earlier results were produced, but they must not be used to block, slow, or
expand current work. In particular, do not insist that the operator rescore candidates, preserve a
blind, perform an acceptance review, verify an operator-provided archive, or restore an abandoned
held-out protocol. Do not call a lightweight result more rigorous than it is, but a plain exploratory
label and an honest sample count are sufficient.

For the operator's own machines and inputs, including 5--10 image extensions, keep only what helps
the experiment: the input set, model/settings, prompt variant, separate output directory, and a short
qualitative or quantitative summary. A simple file count, decode check, or runtime log is optional
when it answers a concrete doubt, not a mandatory ritual. Safety checks for a duplicate GPU process,
destructive deletion, inadequate resources, or accidental overwrite still apply, but keep them
short and targeted. Prefer running and learning from the experiment over building process machinery.

## Historical formal-v1 status: operator-amended reduced replication

The operator explicitly ended the preregistered 300-candidate scoring round and 60-candidate repeat.
The completed human test analysis is a **reduced post-unblinding replication analysis**: 32 sources
(eight per style), all five methods per source, and 160 complete candidate ratings. Sources were
selected after unblinding by prioritizing existing rating completion, with seed `20260822` for ties;
99 ratings were inherited and 61 were completed afterward. Repeat reliability is unavailable.

Never call this the original confirmatory formal-v1 held-out test or ask the operator to restore the
abandoned rounds. It is an exploratory, completion-informed reduced replication and not an
equivalent replacement for the frozen 300-candidate design. Original review materials remain
preserved. The historical amendment, freeze, and results are in
`docs/HANDOFF_FORMAL_V1_HELDOUT_TEST.md`, `docs/results/formal_v1_reduced_32/`, and
`docs/results/formal_v1_reduced_heldout_20260822.md`. This history does not impose an active blind,
visual seal, acceptance gate, or prohibition on new exploratory work directed by the operator.

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
- Pin upstream revisions when it is useful. Record local-file checksums only when the operator
  explicitly requests transfer or integrity verification; do not impose hash work by default.
- Review third-party weight licenses independently of this repository's Apache-2.0 license.
- Do not report copy-backend, smoke-test, mock, downloaded-file, or uncalibrated metric output as a
  research result.
- Run Ruff, pytest, and every new script's `--help` locally before handoff.
- Update `docs/HANDOFF_AUTODL.md` when environment state, model inventory, or next-server commands
  materially change.
- Offer `scripts/package_run.py`, ZIP verification, or SHA-256 only when the operator explicitly
  requests packaging or integrity verification. They are not default completion requirements for
  any experiment. Never recommend broad wildcard deletion of outputs, models, caches, or datasets.

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

## Historical milestone record

The material below records how earlier work was scoped and interpreted. It is useful provenance,
but its freezes, stopping rules, seals, and prescribed next steps are no longer active operator
requirements. The operator policy at the top of this file governs current execution.

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
package. Blinded human calibration and test thresholds/routing rules are now frozen. Test images
remain visually sealed. The active task is to prepare held-out tooling, generate the missing 60
FLUX test outputs once, and conduct the preregistered five-method held-out validation.

A FLUX seed-stability extension completed 20/20 outputs for both seed 43 and seed 44. The complete
seed-44 archive passed CRC, record, and image validation with SHA-256
`9c2042df4b4ee81074467b0cc2f395d044421f719e49a5cb3d84140d5fcce591`. Stop adding seeds. Keep
seeds 43/44 separate from the primary seed-42 evaluation and do not select a favorable seed or use
the extension for parameter tuning.

Formal calibration generation is complete: each of the four frozen SDXL methods has 100/100 new
calibration+test records with zero failures, and FLUX has 40/40 calibration records with zero
failures. All five returned archives and SHA-256 sidecars passed checksum, ZIP CRC, manifest-record,
output-file, decode, mode, and resolution checks locally. Do not generate more formal candidates
now. Preserve test outputs unopened. Calibration-only DINO, CLIP, and paired ArcFace execution is
complete for 200/200 method-source pairs with no evaluator failures; ArcFace yielded 197 cosine
values and three explicit generated-output no-face statuses. One complete 200-pair method-hidden
human round is now frozen and unblinded; FLUX passed 37/40 while each SDXL method passed 1--3/40
under the all-dimensions-at-least-4 rule. Round B was waived before unblinding, so repeat agreement
is unavailable. Failure-type blanks mean unreported, not no failure. The frozen test rule uses FLUX
first and rejects its failures; existing SDXL fallbacks rescued 0/3 FLUX calibration failures. The
held-out hypothesis, outcomes, missing-data rules, single-rater 300-pair plus stratified 20% retest
design, paired statistics, and immutable method settings are preregistered in
`docs/HANDOFF_FORMAL_V1_HELDOUT_TEST.md`. The only remaining formal-v1 generation is the frozen
60-source FLUX test batch. Do not visually inspect any test output before the prescribed blind
materials are built and scored; machine-only integrity validation is required and allowed.

The operator has adopted a narrow private-research interpretation under which ArcFace/InsightFace
may be used only as a paired face-drift diagnostic for these fixed experimental inputs and outputs.
Do not turn it into identity search, identification, authentication, surveillance, enrollment, or a
real-person recognition system; do not publish reusable face templates. Record no-face failures and
the operator interpretation with any result. This operational boundary is not legal advice or a
claim that the license text is unambiguous.

Primary evaluation uses DINOv2 Base, CLIP ViT-L/14, InsightFace where a face is detectable, a
structured Qwen2.5-VL-3B style-removal rubric, and blinded human review. Depth, Refiner, InstantID,
RealVisXL, Florence, 7B VLM auditing, and LoRA training remain secondary.

## Optional research discipline

Use the following practices when they improve the current experiment; they are not automatic gates
and do not reinstate blind testing, formal acceptance, freezing, or statistical requirements.

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

A small result is reportable as an exploratory finding when the real model ran, the input count and
main settings are stated, and the summary honestly describes what was observed and its limitations.
It does not require a frozen split, blind review, calibrated threshold, acceptance test, repeat
rating, formal metric suite, archive audit, or inferential statistics. Do not inflate exploratory
evidence into a large-scale or confirmatory claim, but do not withhold a useful result merely because
formal evaluation machinery was not used.
