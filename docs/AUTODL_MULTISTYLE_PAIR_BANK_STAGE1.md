# AutoDL multistyle pair-bank Stage 1

This run creates only Base FLUX Stage 1 candidates for the curated `candidate` rows. It does not
run held-out sources, sequential Stage 2, a LoRA, or a scorer.

## Transfer prerequisite

Upload the prepared local archive `material_styles_v2-source-bank.zip` to
`/root/autodl-tmp/face-destyle/packages/material_styles_v2-source-bank.zip`, then extract without
overwriting an existing file:

```bash
export FACE_DESTYLE_ROOT=/root/autodl-tmp/face-destyle
export DATA_ROOT="$FACE_DESTYLE_ROOT/data/Face-DeStyle-Data"
export SOURCE_PACKAGE="$FACE_DESTYLE_ROOT/packages/material_styles_v2-source-bank.zip"

test -f "$SOURCE_PACKAGE"
mkdir -p "$DATA_ROOT"
unzip -n "$SOURCE_PACKAGE" -d "$DATA_ROOT"
```

The resulting paths must include:

```text
/root/autodl-tmp/face-destyle/data/Face-DeStyle-Data/extensions/material_styles_v2/raw/clay
/root/autodl-tmp/face-destyle/data/Face-DeStyle-Data/extensions/material_styles_v2/raw/origami
```

The 3D sources already use the existing `raw`, `batch1/raw`, and `batch2/raw` trees.

## Session setup and safe repository sync

Run inside persistent `tmux`. GitHub needs AutoDL's documented academic proxy; unset it immediately
after the pull so local model execution does not inherit stale proxy variables.

```bash
conda activate face-destyle

export FACE_DESTYLE_ROOT=/root/autodl-tmp/face-destyle
export REPO="$FACE_DESTYLE_ROOT/code/Face-DeStyle-Pipeline"
export DATA_ROOT="$FACE_DESTYLE_ROOT/data/Face-DeStyle-Data"
export MODEL_DIR="$FACE_DESTYLE_ROOT/models/diffusion/FLUX.1-Kontext-dev"
export MODEL_MANIFEST="$FACE_DESTYLE_ROOT/models/download-manifests/flux1-kontext-dev-modelscope-master.txt"
export MODEL_HASHES="$FACE_DESTYLE_ROOT/models/download-manifests/flux1-kontext-dev-modelscope-master.config-files.sha256"

export HF_HOME="$FACE_DESTYLE_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="$FACE_DESTYLE_ROOT/cache/torch"
unset OMP_NUM_THREADS

cd "$REPO"
git status --short --branch
source /etc/network_turbo
git pull --ff-only origin main
git log -1 --oneline
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
```

Stop rather than resetting if `git status` shows server-side changes or the fast-forward pull
fails.

## Preflight

```bash
nvidia-smi
python -c 'import torch; print("CUDA available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'
python -c 'from face_destyle.data.pair_bank import load_pair_bank_source_list; print("pair_bank_import=ok")'
python scripts/run_flux_kontext_probe.py --help | grep -- --source-list

test -d "$MODEL_DIR"
test -f "$MODEL_MANIFEST"
test -f "$MODEL_HASHES"
test -f "$DATA_ROOT/extensions/material_styles_v2/raw/clay/matv2-clay-024.png"
test -f "$DATA_ROOT/extensions/material_styles_v2/raw/origami/matv2-origami-030.png"
pgrep -af run_flux_kontext_probe.py || true
```

If the new Python module is not visible from the existing editable environment, run only:

```bash
python -m pip install -e . --no-deps
```

Do not install or upgrade GPU dependencies.

## 3D Stage 1: 27 candidates

```bash
set +e
export RUN="$FACE_DESTYLE_ROOT/outputs/pair-bank-3d-v1/stage1"

if test -e "$RUN"; then
  echo "STOP: output path already exists: $RUN"
else
  mkdir -p "$RUN/images"
  python scripts/run_flux_kontext_probe.py \
    --source-list data/manifests/multistyle-pair-bank/3d_cartoon_sources.csv \
    --data-root "$DATA_ROOT" \
    --required-style 3d_cartoon \
    --probe-stage batch \
    --prompt-stage stage1 \
    --model-dir "$MODEL_DIR" \
    --download-manifest "$MODEL_MANIFEST" \
    --hash-manifest "$MODEL_HASHES" \
    --output-dir "$RUN/images" \
    --records-output "$RUN/records.jsonl" \
    --failures-output "$RUN/failures.jsonl" \
    --styles-config configs/styles_3d_lora.yaml \
    --seed 42 \
    --num-inference-steps 28 \
    --guidance-scale 2.5 2>&1 | tee "$RUN/generate.log"
  runner_rc=${PIPESTATUS[0]}
  echo "PAIR_BANK_3D_STAGE1_EXIT_CODE=$runner_rc"
fi
```

Run only one style at a time. Confirm the first exit code is zero before continuing.

## Clay Stage 1: 19 candidates

```bash
set +e
export RUN="$FACE_DESTYLE_ROOT/outputs/pair-bank-clay-v1/stage1"

if test -e "$RUN"; then
  echo "STOP: output path already exists: $RUN"
else
  mkdir -p "$RUN/images"
  python scripts/run_flux_kontext_probe.py \
    --source-list data/manifests/multistyle-pair-bank/clay_sources.csv \
    --data-root "$DATA_ROOT" \
    --required-style clay \
    --probe-stage batch \
    --prompt-stage stage1 \
    --model-dir "$MODEL_DIR" \
    --download-manifest "$MODEL_MANIFEST" \
    --hash-manifest "$MODEL_HASHES" \
    --output-dir "$RUN/images" \
    --records-output "$RUN/records.jsonl" \
    --failures-output "$RUN/failures.jsonl" \
    --styles-config configs/styles.yaml \
    --seed 42 \
    --num-inference-steps 28 \
    --guidance-scale 2.5 2>&1 | tee "$RUN/generate.log"
  runner_rc=${PIPESTATUS[0]}
  echo "PAIR_BANK_CLAY_STAGE1_EXIT_CODE=$runner_rc"
fi
```

## Origami Stage 1: 24 candidates

```bash
set +e
export RUN="$FACE_DESTYLE_ROOT/outputs/pair-bank-origami-v1/stage1"

if test -e "$RUN"; then
  echo "STOP: output path already exists: $RUN"
else
  mkdir -p "$RUN/images"
  python scripts/run_flux_kontext_probe.py \
    --source-list data/manifests/multistyle-pair-bank/origami_sources.csv \
    --data-root "$DATA_ROOT" \
    --required-style origami \
    --probe-stage batch \
    --prompt-stage stage1 \
    --model-dir "$MODEL_DIR" \
    --download-manifest "$MODEL_MANIFEST" \
    --hash-manifest "$MODEL_HASHES" \
    --output-dir "$RUN/images" \
    --records-output "$RUN/records.jsonl" \
    --failures-output "$RUN/failures.jsonl" \
    --styles-config configs/styles.yaml \
    --seed 42 \
    --num-inference-steps 28 \
    --guidance-scale 2.5 2>&1 | tee "$RUN/generate.log"
  runner_rc=${PIPESTATUS[0]}
  echo "PAIR_BANK_ORIGAMI_STAGE1_EXIT_CODE=$runner_rc"
fi
```

Expected success counts are 27, 19, and 24. After each run, check `records.jsonl`, the image count,
and any non-empty `failures.jsonl`. Download the three run directories before deciding which
sources need a sequential second pass. Do not run Stage 2 automatically across all 70 sources.
