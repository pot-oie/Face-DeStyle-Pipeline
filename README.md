# Face-DeStyle-Pipeline

A compact, reproducible research scaffold for turning stylized face images into natural-looking
photographs while retaining identity, pose, composition, and background structure.

## Start here

Agents and contributors should read [`AGENTS.md`](AGENTS.md) first. The detailed research contract
is in [`docs/research_context.md`](docs/research_context.md), and the approved process for finding,
licensing, screening, and splitting artistic images is in
[`docs/data_acquisition.md`](docs/data_acquisition.md).

> “This repository is a compact and independent reproduction developed from an undergraduate
> innovation project under the same research direction as DeStyle. It focuses on face-domain
> destylization, structural conditioning, and quality filtering. It is not the official
> implementation of DeStyle-350K.”

## Research questions and scope

The planned 1–2 week study asks whether a two-stage destylization pipeline benefits from
style-adaptive prompts and structural controls. It compares generic versus category-adaptive
prompts, prompt-only generation, whole-image Canny, face/background region-aware Canny, and an
optional pose condition. Results will be judged on content preservation, identity preservation,
style removal, and human pass rate. A dual-threshold gate produces only accepted
`<destylized content, style reference, original style target>` triplets.

This repository owns code, small public examples, configuration, tests, and documentation. It
does not contain DeStyle-350K, model weights, full datasets, private faces, caches, checkpoints, or
bulk outputs. It is an independent undergraduate reproduction in the same research direction,
not an official DeStyle or DeStyle-350K implementation.

## Implemented versus planned

Implemented: strict Pydantic records, JSONL validation, stable metadata IDs, OpenCV Canny,
manual/center masks for smoke testing, a no-op copy backend, a prompt-only SDXL image-to-image
baseline for AutoDL, explicit pixel similarity, dual-threshold filtering, deterministic triplet
sampling, CSV export, and unit tests. Diffusers tests inject a mock pipeline and never download a
model.

Declared for AutoDL, but not yet GPU-verified: ControlNet conditioning, production face parsing,
pose/depth extraction, DINO/CLIP/SigLIP content or semantic similarity, ArcFace identity
similarity, and VLM style-removal scoring. `configs/models.yaml` records expected server assets and
licenses; `configs/experiments.yaml` declares primary and extension comparisons. The copy backend
is only plumbing validation and is not an experimental result. The first Diffusers baseline is
prompt-only and must not be described as a controlled or evaluated method.

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

The first GPU baseline uses `stabilityai/stable-diffusion-xl-base-1.0` at revision
`462165984030d82259a11f4367a4eed129e94a7b` through `AutoPipelineForImage2Image`. It performs
prompt-only SDXL img2img: no Canny, ControlNet, face mask, pose, refiner, or research metric is
applied. Read [`docs/autodl_setup.md`](docs/autodl_setup.md) before downloading weights.

Single-image example:

```bash
python scripts/run_destylization.py \
  --input /path/to/authorized_comic_face.png \
  --style-category comic \
  --record-id demo-001 \
  --backend diffusers \
  --config configs/inference.yaml \
  --styles-config configs/styles.yaml \
  --output-dir /root/autodl-tmp/face-destyle/outputs/single \
  --records-output /root/autodl-tmp/face-destyle/outputs/single/record.jsonl
```

For JSONL batch input, replace `--input`, `--style-category`, and `--record-id` with
`--metadata /path/to/metadata.jsonl`. Output locations always come from CLI arguments.

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

## Planned evaluation and ablations

Formal evaluation will calibrate thresholds against a human-annotated validation set and report:

- DINO/CLIP content preservation;
- ArcFace identity preservation;
- VLM style removal;
- human acceptance rate.

The planned factorial comparisons are generic versus adaptive prompts; prompt-only versus global
Canny versus face/background region-aware Canny; and the optional addition of pose control. Seeds,
model revisions, prompts, control strengths, and threshold calibration must be recorded.

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
