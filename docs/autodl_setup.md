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
export TORCH_HOME=/data/cache/torch
export PIP_CACHE_DIR=/data/cache/pip
```

Clone once or pull subsequent code updates, then install optional GPU dependencies:

```bash
git clone <your-repository-url> ~/code/Face-DeStyle-Pipeline
cd ~/code/Face-DeStyle-Pipeline
python -m pip install -e ".[gpu,dev]"
python scripts/check_environment.py --gpu
```

Confirm the chosen PyTorch build matches the server CUDA/driver combination. This repository does
not download models; use the model provider's documented authentication and pin exact revisions.

## Synchronization and backup

Do not commit weights, Hugging Face caches, checkpoints, raw/processed datasets, or bulk outputs.
Synchronize only code, configs, summary CSV files, documentation, and a few authorized public
examples. Git is not a data backup. Important raw data, metadata, outputs, and checkpoints require a
separate backup with access controls and restoration tests.
