# AutoDL setup

This document contains durable, non-secret AutoDL operating context. Root-level `AGENTS.md` gives
coding agents the same environment boundary. Current machine allocation, live proxy health, SSH
addresses, tokens, and signed download URLs are session state and must not be committed.

## Recommended layout

Keep source code separate from persistent bulk storage. Adapt mount names to the purchased image:

```text
~/code/Face-DeStyle-Pipeline/       Git checkout
/data/face-destyle/models/          downloaded model weights
/data/face-destyle/datasets/        licensed/private datasets
/data/face-destyle/outputs/         bulk generations
/data/face-destyle/checkpoints/     training checkpoints, if later required
/data/cache/huggingface/            Hugging Face cache
/data/cache/torch/                  Torch cache
/data/cache/pip/                    pip cache
```

Never hard-code these paths in source; pass them through CLI arguments, config, or environment.

## Environment

Create a Conda environment with CPython 3.10, activate it, then verify that the server driver is
visible before installing model code:

```bash
conda create -n face-destyle python=3.10 -y
conda activate face-destyle
nvidia-smi
python -c "import sys; print(sys.version)"
```

Point caches at the persistent data disk (replace `/data` if the instance uses another mount):

```bash
export HF_HOME=/data/cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TORCH_HOME=/data/cache/torch
export PIP_CACHE_DIR=/data/cache/pip
```

## Network routes in mainland China

AutoDL may need an acceleration route for GitHub or Hugging Face. Treat route selection as an
operational decision, not a permanent application setting:

```bash
# AutoDL documented academic proxy for an interactive shell
source /etc/network_turbo

# Cancel it when it slows or breaks ordinary access
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
```

The public Hugging Face mirror is a separate route:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export HF_ENDPOINT=https://hf-mirror.com
```

Do not assume either route is currently healthy, and do not combine them accidentally. Large-file
downloads should be resumable and target the data disk. The current server used HFD/aria2 for
selected Hugging Face repositories and ModelScope for Qwen after direct Xet transfers proved too
slow. Never commit access tokens, credentials, signed CDN URLs, or SSH connection information.

Clone once or pull subsequent code updates, then install optional GPU dependencies:

```bash
git clone <your-repository-url> ~/code/Face-DeStyle-Pipeline
cd ~/code/Face-DeStyle-Pipeline
python -m pip install -e ".[gpu,dev]"
python scripts/check_environment.py --gpu
python scripts/check_model_assets.py --config configs/models.yaml
```

The optional evaluation runtime is intentionally separate from the first generation baseline:

```bash
python -m pip install -e ".[gpu,evaluation,dev]"
```

It adds InsightFace, ONNX Runtime GPU, and Qwen vision helpers. Install it when implementing the
formal evaluators, not merely to check model files.

Confirm the chosen PyTorch build matches the server CUDA/driver combination. This repository does
not download models; use the model provider's documented authentication and pin exact revisions.

## First prompt-only GPU baseline

The selected baseline is
[`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0),
pinned to Hugging Face revision `462165984030d82259a11f4367a4eed129e94a7b`. The model card
describes SDXL base as capable of generating and modifying images. Its model license is
[CreativeML Open RAIL++-M](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/462165984030d82259a11f4367a4eed129e94a7b/LICENSE.md),
not this repository's Apache-2.0 code license. Review the full model license and use restrictions
before use or redistribution.

Resource planning values below are estimates, not measurements from this repository:

- the repository exposes a 6.94 GB full checkpoint; Diffusers component cache and download
  overhead should be budgeted at roughly 7–10 GB;
- 768×768, batch-size 1 inference in FP16/BF16 should initially budget roughly 12–18 GB VRAM;
- an RTX 3090-class 24 GB GPU is a reasonable target, but verify actual peak allocated/reserved
  memory on the selected image and library versions before increasing resolution or batch size.

The backend requires either `HF_HUB_CACHE` (the preferred explicit setting) or `HF_HOME`. The
following keeps all caches together on persistent storage. AutoDL images sometimes supply a
malformed `OMP_NUM_THREADS`; unset it before Python starts instead of changing it inside the model
backend:

```bash
export FACE_DESTYLE_ROOT=/root/autodl-tmp/face-destyle
unset OMP_NUM_THREADS
export HF_HOME="$FACE_DESTYLE_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="$FACE_DESTYLE_ROOT/cache/torch"
export PIP_CACHE_DIR="$FACE_DESTYLE_ROOT/cache/pip"

python scripts/check_environment.py --gpu
export SOURCE_IMAGE=/path/to/qc-accepted-comic-face.png
test -f "$SOURCE_IMAGE"
python scripts/run_destylization.py \
  --input "$SOURCE_IMAGE" \
  --style-category comic \
  --prompt-mode adaptive \
  --record-id demo-001 \
  --backend diffusers \
  --output-dir "$FACE_DESTYLE_ROOT/outputs/prompt-only" \
  --records-output "$FACE_DESTYLE_ROOT/outputs/prompt-only/records.jsonl"
```

The backend defaults to `local_files_only: true` and loads the pinned snapshot already present in
`HF_HOME`; it fails instead of downloading if that snapshot is incomplete. Run one authorized image
first, monitor `nvidia-smi`, inspect the output manually, and record peak memory. Do not start a
batch until that check succeeds. This baseline does not constitute a research result.

Do not choose a comparison demo arbitrarily. Use a QC-accepted source whose declared category
matches its visible style, retain its stable `source_id`, and verify its recorded SHA-256 after
transfer. Store the image on the persistent data disk, not in Git. Prompt-only and Canny comparisons
must use the same transferred file and settings.

The model registry resolves the pinned Hugging Face snapshot directory and passes that local path
directly to Diffusers. Do not copy `configs/inference.yaml` or replace its model setting with a
host-specific path. `model_asset: sdxl_base` is resolved through `configs/models.yaml`.

After the prompt-only smoke test succeeds, the matched global Canny smoke test is:

```bash
python scripts/check_model_assets.py --asset sdxl_base --asset canny_controlnet
python scripts/run_destylization.py \
  --input "$SOURCE_IMAGE" \
  --style-category comic \
  --prompt-mode adaptive \
  --record-id demo-canny-001 \
  --backend canny \
  --control-scale 0.8 \
  --seed 42 \
  --output-dir "$FACE_DESTYLE_ROOT/outputs/global-canny/images" \
  --records-output "$FACE_DESTYLE_ROOT/outputs/global-canny/records.jsonl"
```

Inspect `demo-canny-001.canny.png` as well as the generated image. This backend applies global
edges; it does not claim face-region awareness or validated quality improvement.

For a frozen external dataset, set its root separately from model storage:

```bash
export FACE_DESTYLE_DATA_ROOT=/root/autodl-tmp/face-destyle/data/Face-DeStyle-Data
python scripts/run_destylization.py \
  --manifest data/manifests/formal-v1/inputs.jsonl \
  --split pilot \
  --backend diffusers \
  --output-dir "$FACE_DESTYLE_ROOT/outputs/prompt-adaptive" \
  --records-output "$FACE_DESTYLE_ROOT/outputs/prompt-adaptive/records.jsonl"
```

The manifest is safe to commit only after sanitization and split freeze. Its image paths must be
relative to `FACE_DESTYLE_DATA_ROOT`; raw images remain on the data disk.

Before that freeze, a rich private provenance manifest can be converted into a temporary strict
runtime manifest. The converter keeps only accepted records in the requested split, derives paths
below the configured anchor (normally `raw/`), and verifies every SHA-256 against the transferred
data root. It does not modify the provenance source or create a formal release manifest:

```bash
python scripts/build_runtime_manifest.py \
  --input "$FACE_DESTYLE_DATA_ROOT/manifests/pilot_manifest.jsonl" \
  --data-root "$FACE_DESTYLE_DATA_ROOT" \
  --output /tmp/face-destyle-pilot-runtime.jsonl
```

For exploratory pilot tuning, one process can reuse a loaded pipeline across several img2img
strengths. Each strength still receives a separate image directory and `records.jsonl`; the sweep
root also retains the exact runtime manifest and a small sweep declaration. Do not use this command
to tune on calibration or test data:

```bash
python scripts/run_strength_sweep.py \
  --manifest /tmp/face-destyle-pilot-runtime.jsonl \
  --data-root "$FACE_DESTYLE_DATA_ROOT" \
  --output-root "$FACE_DESTYLE_ROOT/outputs/pilot-strength-sweep" \
  --backend diffusers \
  --prompt-mode adaptive \
  --strength 0.50 0.60 0.70 0.80 \
  --seed 42
```

For isolated runs, `run_destylization.py` also accepts `--strength`, `--guidance-scale`, and
`--num-inference-steps` overrides for the Diffusers and Canny backends. Change one factor at a time
when interpreting a comparison.

## Package, download, and clean a completed run

Package each completed experiment before releasing or cloning the GPU instance. Put the archive
outside the run directory. `--cleanup` is deliberately limited to one child of the configured
outputs root and runs only after the ZIP passes an integrity test and its SHA-256 file is written:

```bash
export RUN_DIR="$FACE_DESTYLE_ROOT/outputs/matched-synthetic-comic-001-seed-42"
export ARCHIVE_DIR="$FACE_DESTYLE_ROOT/exports"
export ARCHIVE="$ARCHIVE_DIR/$(basename "$RUN_DIR").zip"

python scripts/package_run.py \
  --run-dir "$RUN_DIR" \
  --archive "$ARCHIVE" \
  --cleanup

cd "$ARCHIVE_DIR"
sha256sum -c "$(basename "$ARCHIVE").sha256"
test ! -e "$RUN_DIR"
```

Download both the `.zip` and `.zip.sha256` files. The archive contains paths relative to the run
directory and preserves images, condition images, and JSONL records. The script rejects symlinks,
existing archives, paths outside `outputs`, the entire `outputs` root, and archives placed inside
the run being cleaned. It never touches models, caches, datasets, or other runs.

The purchased-data-disk layout used by the current inventory is represented in
`configs/models.yaml`. Local paths are relative to `FACE_DESTYLE_ROOT`; cached paths use `HF_HOME`.
See `docs/model_assets.md` for license notes and `docs/HANDOFF_AUTODL.md` for the next GPU session.

## Synchronization and backup

Do not commit weights, Hugging Face caches, checkpoints, raw/processed datasets, or bulk outputs.
Synchronize only code, configs, summary CSV files, documentation, and a few authorized public
examples. Git is not a data backup. Important raw data, metadata, outputs, and checkpoints require a
separate backup with access controls and restoration tests.
