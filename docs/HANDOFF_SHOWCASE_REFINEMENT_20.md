# AutoDL handoff: 20-source showcase refinement

This is a display-oriented lightweight inference extension, not a reopening of the finalized
137-source routing validation. It runs 10 selected 3D-cartoon sources and 10 selected needle-felt
sources through two sequential FLUX.1 Kontext edits, producing up to 40 candidate images. It does
not train or load a LoRA, and its outcomes must not be added to the finalized study counts.

The 3D route focuses on realistic facial anatomy followed by removal of residual rendered and
plastic cues. The needle-felt route is intentionally a portrait-oriented semantic reconstruction:
it may replace a felt bust edge, pedestal, or support with plausible human shoulders, torso, and
clothing. Judge those images as showcase candidates rather than strict full-frame material removal.

## 1. Enter the existing environment

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate face-destyle

export FACE_DESTYLE_ROOT=/root/autodl-tmp/face-destyle
export REPO="$FACE_DESTYLE_ROOT/code/Face-DeStyle-Pipeline"
export DATA_ROOT="$FACE_DESTYLE_ROOT/data/Face-DeStyle-Data"
export MODEL_DIR="$FACE_DESTYLE_ROOT/models/diffusion/FLUX.1-Kontext-dev"
export MODEL_MANIFEST="$FACE_DESTYLE_ROOT/models/download-manifests/flux1-kontext-dev-modelscope-master.txt"
export MODEL_HASHES="$FACE_DESTYLE_ROOT/models/download-manifests/flux1-kontext-dev-modelscope-master.config-files.sha256"
export RUN_ROOT="$FACE_DESTYLE_ROOT/outputs/showcase-refinement-20-v1"

cd "$REPO"
source /etc/network_turbo
git pull --ff-only origin main
unset http_proxy
unset https_proxy
```

The model and data checks from the preceding routing run are sufficient. Do not repeat expensive
weight hashing or create output hash inventories for this display extension.

## 2. Run Stage 1

Run inside `screen` so the job survives a disconnected terminal.

```bash
mkdir -p "$RUN_ROOT/stage1/images"
set -o pipefail

python scripts/run_flux_kontext_probe.py \
  --manifest data/manifests/multistyle-routing/showcase_refinement_20.jsonl \
  --data-root "$DATA_ROOT" \
  --split extension \
  --required-style 3d_cartoon \
  --required-style needle_felt \
  --probe-stage batch \
  --prompt-stage stage1 \
  --model-dir "$MODEL_DIR" \
  --download-manifest "$MODEL_MANIFEST" \
  --hash-manifest "$MODEL_HASHES" \
  --output-dir "$RUN_ROOT/stage1/images" \
  --records-output "$RUN_ROOT/stage1/records.jsonl" \
  --failures-output "$RUN_ROOT/stage1/failures.jsonl" \
  --styles-config configs/styles_showcase_refinement.yaml \
  --seed 42 \
  --num-inference-steps 28 \
  --guidance-scale 2.5 \
  2>&1 | tee "$RUN_ROOT/stage1/generate.log"

echo "SHOWCASE_STAGE1_EXIT_CODE=${PIPESTATUS[0]}"
```

Stage 1 is ready for continuation when `records.jsonl` contains 20 lines and failures are empty.
If the process was interrupted, rerun the same command with `--resume`; do not delete successful
images or records.

## 3. Run true sequential Stage 2

Only start this after Stage 1 has 20 successes. Stage 2 reads Stage 1 outputs, not the original
source manifest.

```bash
mkdir -p "$RUN_ROOT/stage2-sequential/images"
set -o pipefail

python scripts/run_flux_kontext_probe.py \
  --input-records "$RUN_ROOT/stage1/records.jsonl" \
  --required-style 3d_cartoon \
  --required-style needle_felt \
  --probe-stage batch \
  --prompt-stage stage2 \
  --model-dir "$MODEL_DIR" \
  --download-manifest "$MODEL_MANIFEST" \
  --hash-manifest "$MODEL_HASHES" \
  --output-dir "$RUN_ROOT/stage2-sequential/images" \
  --records-output "$RUN_ROOT/stage2-sequential/records.jsonl" \
  --failures-output "$RUN_ROOT/stage2-sequential/failures.jsonl" \
  --styles-config configs/styles_showcase_refinement.yaml \
  --seed 42 \
  --num-inference-steps 28 \
  --guidance-scale 2.5 \
  2>&1 | tee "$RUN_ROOT/stage2-sequential/generate.log"

echo "SHOWCASE_STAGE2_EXIT_CODE=${PIPESTATUS[0]}"
```

If interrupted, rerun the same Stage 2 command with `--resume`.

## 4. Compact completion check and download

```bash
wc -l \
  "$RUN_ROOT/stage1/records.jsonl" \
  "$RUN_ROOT/stage2-sequential/records.jsonl"

find "$RUN_ROOT" -name failures.jsonl -type f -size +0 -print
du -sh "$RUN_ROOT"
```

Download the complete `showcase-refinement-20-v1` directory for local visual review. Do not copy
the generated image directories, model weights, or large archives into Git.
