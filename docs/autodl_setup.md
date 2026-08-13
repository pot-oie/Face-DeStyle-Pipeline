# AutoDL setup

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

The backend refuses to load real weights unless `HF_HOME` is set. If
`HUGGINGFACE_HUB_CACHE` is supplied, it must be inside `HF_HOME`:

```bash
export FACE_DESTYLE_ROOT=/root/autodl-tmp/face-destyle
export HF_HOME="$FACE_DESTYLE_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="$FACE_DESTYLE_ROOT/cache/torch"
export PIP_CACHE_DIR="$FACE_DESTYLE_ROOT/cache/pip"

python scripts/check_environment.py --gpu
python scripts/run_destylization.py \
  --input "$FACE_DESTYLE_ROOT/data/authorized/demo.png" \
  --style-category comic \
  --record-id demo-001 \
  --backend diffusers \
  --output-dir "$FACE_DESTYLE_ROOT/outputs/prompt-only" \
  --records-output "$FACE_DESTYLE_ROOT/outputs/prompt-only/records.jsonl"
```

The backend defaults to `local_files_only: true` and loads the pinned snapshot already present in
`HF_HOME`; it fails instead of downloading if that snapshot is incomplete. Run one authorized image
first, monitor `nvidia-smi`, inspect the output manually, and record peak memory. Do not start a
batch until that check succeeds. This baseline does not constitute a research result.

The purchased-data-disk layout used by the current inventory is represented in
`configs/models.yaml`. Local paths are relative to `FACE_DESTYLE_ROOT`; cached paths use `HF_HOME`.
See `docs/model_assets.md` for license notes and `docs/HANDOFF_AUTODL.md` for the next GPU session.

## Synchronization and backup

Do not commit weights, Hugging Face caches, checkpoints, raw/processed datasets, or bulk outputs.
Synchronize only code, configs, summary CSV files, documentation, and a few authorized public
examples. Git is not a data backup. Important raw data, metadata, outputs, and checkpoints require a
separate backup with access controls and restoration tests.
