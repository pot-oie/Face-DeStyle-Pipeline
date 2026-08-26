# Face-DeStyle-Pipeline

A compact, reproducible research scaffold for turning stylized face images into natural-looking
photographs while retaining identity, pose, composition, and background structure.

## Start here

Agents and contributors should read [`AGENTS.md`](AGENTS.md) first. The detailed research contract
is in [`docs/research_context.md`](docs/research_context.md), and the approved process for finding,
licensing, screening, and splitting artistic images is in
[`docs/data_acquisition.md`](docs/data_acquisition.md).

The style-specific LoRA branch and multistyle processing-route work are complete. Start with the
[`final multistyle report`](docs/results/multistyle_project_closure_20260826.md) and
[`experiment evidence index`](docs/EXPERIMENT_EVIDENCE_INDEX.md). There is no active AutoDL run or
unfinished experiment.

> “This repository is a compact and independent reproduction developed from an undergraduate
> innovation project under the same research direction as DeStyle. It focuses on face-domain
> destylization, structural conditioning, and quality filtering. It is not the official
> implementation of DeStyle-350K.”

## Research outcome and scope

The completed study tested style-adaptive prompts, structural controls, original-BF16 FLUX Kontext,
material-aware sequential editing, and small failure-targeted LoRAs. SDXL prompt/Canny variants
reached a practical plateau, while FLUX showed a stronger signal on comic, ink, and watercolor.
Geometry/material-entangled styles remained harder. Origami V1 checkpoint 100 is retained as a
limited adapter at about 3/6 strict holdout passes; Origami V2/V2.1 and the eight-pair 3D LoRA did
not improve the selected boundary. The final output is an evidence-backed per-style route, not a
universal multistyle adapter.

This repository owns code, small public examples, configuration, tests, and documentation. It
does not contain DeStyle-350K, model weights, full datasets, private faces, caches, checkpoints, or
bulk outputs. It is an independent undergraduate reproduction in the same research direction,
not an official DeStyle or DeStyle-350K implementation.

Portable checksum-pinned split manifests may be source-controlled under `data/manifests/`; their
raw images resolve against a separate data root and never enter Git. This keeps code and experiment
identity reproducible without using Git history as a private/bulk-image backup.

Development is split between a lightweight local macOS environment and an AutoDL data-disk/GPU
environment. Durable agent rules, including mainland mirror/proxy cautions, are in `AGENTS.md`;
server setup is in `docs/autodl_setup.md`. Credentials and live connection details are never stored
in the repository.

## Implemented and evaluated

Implemented: strict Pydantic records, JSONL validation, stable metadata IDs, a human-reviewed
manifest/record processing router, OpenCV Canny,
manual/center masks for smoke testing, a no-op copy backend, a prompt-only SDXL image-to-image
baseline for AutoDL, global and face-parsing-aware region Canny ControlNet backends with saved
condition artifacts, an original-BF16 FLUX.1 Kontext native-1024 prompt-editing backend with model
offload and resumable records, explicit pixel similarity, dual-threshold filtering, deterministic
triplet sampling, CSV export, and unit tests. Prompt-only, global Canny, region-aware Canny, and the
exploratory FLUX probe have completed 20-source pilot generation; this is not formal metric or
quality validation.
Local Diffusers tests inject mock pipelines and never download a model.

The AutoDL-only formal evaluator ran on the primary pilot with DINOv2, CLIP, paired ArcFace drift
diagnostics, and a structured Qwen2.5-VL rubric. The human test was ultimately completed as a
32-source, five-method reduced post-unblinding replication, not the abandoned confirmatory design;
see [`docs/results/formal_v1_reduced_heldout_20260822.md`](docs/results/formal_v1_reduced_heldout_20260822.md).
Raw cosines and rubric scores are diagnostics, not acceptance probabilities.

Historical method-hidden review tooling and its private unblinding material remain preserved. The
operator ended the planned 300-candidate test and repeat round; they must not be restarted or
described as unfinished requirements.

Pose/depth extraction and SigLIP remain unverified extensions. `configs/models.yaml` records
expected server assets and licenses; `configs/experiments.yaml` preserves the declared comparison
matrix. The copy backend is only plumbing validation and is not an experimental result. Historical
GPU commands reproduce earlier work but do not authorize a new run.

## Installation

Use CPython 3.10. The normal local install has no Torch or GPU stack:

```bash
python -m pip install -e ".[dev]"
```

On an AutoDL GPU host, after selecting a CUDA-compatible PyTorch build:

```bash
python -m pip install -e ".[gpu,dev]"
```

No command in this repository downloads a model automatically. See
[`docs/autodl_setup.md`](docs/autodl_setup.md).

To audit already-downloaded server assets without importing Torch or contacting a model host:

```bash
python scripts/check_model_assets.py --config configs/models.yaml
python scripts/list_experiments.py --seed 42 --json
```

## Quick Start: local no-GPU smoke test

```bash
python scripts/check_environment.py
python scripts/prepare_data.py --input-dir data/samples --style-category comic \
  --output outputs/metadata.jsonl
python scripts/run_destylization.py --metadata outputs/metadata.jsonl \
  --output-dir outputs/copied --records-output outputs/destylized.jsonl
python scripts/evaluate_pairs.py --records outputs/destylized.jsonl \
  --output outputs/evaluations.jsonl
python scripts/filter_pairs.py --evaluations outputs/evaluations.jsonl \
  --output outputs/filtered.jsonl
python scripts/build_triplets.py --evaluations outputs/filtered.jsonl \
  --output outputs/triplets.jsonl --seed 42 -k 1
```

`smoke_test_similarity` is a normalized pixel similarity, never DINO, ArcFace, CLIP, or VLM. The
smoke evaluator uses an explicitly labeled, unmeasured style-removal sentinel solely so all stages
can be exercised. Do not report it as a scientific metric.

## First AutoDL Diffusers baseline

This section is a historical reproduction example, not an active experiment instruction.

The first GPU baseline uses `stabilityai/stable-diffusion-xl-base-1.0` at revision
`462165984030d82259a11f4367a4eed129e94a7b` through `AutoPipelineForImage2Image`. It performs
prompt-only SDXL img2img: no Canny, ControlNet, face mask, pose, refiner, or research metric is
applied. Read [`docs/autodl_setup.md`](docs/autodl_setup.md) before downloading weights.

Single-image example:

```bash
python scripts/run_destylization.py \
  --input /path/to/authorized_comic_face.png \
  --style-category comic \
  --prompt-mode adaptive \
  --record-id demo-001 \
  --backend diffusers \
  --config configs/inference.yaml \
  --styles-config configs/styles.yaml \
  --output-dir /root/autodl-tmp/face-destyle/outputs/single \
  --records-output /root/autodl-tmp/face-destyle/outputs/single/record.jsonl
```

For JSONL batch input, replace `--input`, `--style-category`, and `--record-id` with
`--metadata /path/to/metadata.jsonl`. Output locations always come from CLI arguments.

The first structural comparison uses the registered global Canny ControlNet and writes the exact
conditioning image beside the generated image:

```bash
python scripts/run_destylization.py \
  --input /path/to/authorized_comic_face.png \
  --style-category comic \
  --prompt-mode adaptive \
  --record-id demo-canny-001 \
  --backend canny \
  --control-scale 0.4 \
  --output-dir /root/autodl-tmp/face-destyle/outputs/canny \
  --records-output /root/autodl-tmp/face-destyle/outputs/canny/record.jsonl
```

This is global edge conditioning only. The separate `region_canny` backend uses the registered face
parser, keeps parsed head-region edges at full strength, and weakens background edges. Its saved
face mask and composite condition passed a one-source visual smoke check. A later fixed-config
20-source pilot completed without runtime failures but did not show a stable visual improvement
over global Canny; see the AutoDL handoff for the evidence and claim boundary.

For a frozen portable manifest whose images live outside the code repository:

```bash
export FACE_DESTYLE_DATA_ROOT=/path/to/Face-DeStyle-Data
python scripts/run_destylization.py \
  --manifest data/manifests/formal-v1/inputs.jsonl \
  --split pilot \
  --backend diffusers \
  --output-dir /path/to/outputs/prompt-adaptive \
  --records-output /path/to/outputs/prompt-adaptive/records.jsonl
```

Manifest paths are relative to `FACE_DESTYLE_DATA_ROOT` and are checksum-verified before inference.
The loader rejects source groups assigned across multiple splits. Raw data and private provenance
remain outside Git; see [`data/manifests/README.md`](data/manifests/README.md).

After a completed server run, `scripts/package_run.py` creates and verifies a ZIP and SHA-256 file.
Its optional `--cleanup` removes only the selected run directory after verification; see the AutoDL
setup guide for the exact handoff workflow.

## Data formats

Every stage uses one validated JSON object per line. Core types are `ImageRecord`,
`DestylizationRecord`, `EvaluationRecord`, and `TripletRecord` in
`src/face_destyle/schemas.py`. A source example is in `data/metadata.example.jsonl`. Triplets store:

```text
destylized_content_path, style_reference_path, original_style_target_path,
style_category, target_source_id, reference_source_id
```

Only accepted samples are used. Reference and target share a style category but cannot share a
`source_id`.

## Completed evaluation and routing

The repository records:

- DINO/CLIP content preservation;
- paired ArcFace face-drift diagnostics with explicit no-face handling;
- structured VLM style-removal output;
- human strict-pass decisions and limitations.

The final route accepts one-stage Base FLUX for Comic and Watercolor, uses a review-triggered second
edit for Ink, and records explicit fallback/failure handling for Needle-felt. Origami retains the
optional frozen V1 checkpoint 100 plus a limited residual edit; Clay and 3D cartoon retain
teacher/failure handling. Exact evidence and limitations are in the closure report. No new LoRA,
selector, formal rerun, or GPU experiment is active.

## Repository layout

```text
configs/                inference, styles, model registry, and experiment declarations
scripts/                one CLI per pipeline stage
src/face_destyle/       schemas, pipelines, controls, metrics, filtering, and data logic
data/                   documentation, ignored private data, and tiny local samples
docs/                   method, protocol, limitations, AutoDL setup, experiment plan
tests/                  lightweight unit tests
outputs/, results/      ignored generated artifacts, except placeholders
```

## Privacy, copyright, and licensing

Do not commit faces without documented authorization and an appropriate research-use basis. Respect
dataset terms, likeness/privacy rights, copyright, and removal requests. Generated outputs can still
contain identifying information. Code is licensed under Apache-2.0; that license does not grant
rights to datasets, model weights, faces, trademarks, or third-party outputs.
