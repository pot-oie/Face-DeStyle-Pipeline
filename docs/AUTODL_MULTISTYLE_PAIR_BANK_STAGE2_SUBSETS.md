# AutoDL reviewed sequential Stage 2 subsets

The 2026-08-24 local visual review found all 70 Stage 1 jobs technically complete, but only 14
Origami outputs were plausible provisional targets. Clay retained its sculpted material almost
universally, and 3D retained CGI/animation cues. The full routing table is
`docs/results/multistyle_pair_bank_stage1_routing_20260824.csv`.

This run is deliberately limited to 8 3D, 12 Clay, and 10 Origami records. Every Stage 2 input is
the corresponding Stage 1 output named by the original Stage 1 record, not the styled source.
Do not run the other records automatically and do not start LoRA training after generation.

## Session setup and repository sync

Run inside the existing persistent `tmux` session:

```bash
conda activate face-destyle

export FACE_DESTYLE_ROOT=/root/autodl-tmp/face-destyle
export REPO="$FACE_DESTYLE_ROOT/code/Face-DeStyle-Pipeline"
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
git fetch origin main
git merge --ff-only FETCH_HEAD
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

python scripts/run_flux_kontext_probe.py --help | grep -- --source-id
nvidia-smi
python -c 'import torch; print("CUDA available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'
```

Stop if the repository has server-side edits, the pull is not a fast-forward, the new
`--source-id` option is absent, or CUDA is unavailable.

## 3D representative subset: 8

```bash
set +e
export RUN_ROOT="$FACE_DESTYLE_ROOT/outputs/pair-bank-3d-v1"
export RUN="$RUN_ROOT/stage2-sequential"

if test -e "$RUN"; then
  echo "STOP: output path already exists: $RUN"
else
  mkdir -p "$RUN/images"
  python scripts/run_flux_kontext_probe.py \
    --input-records "$RUN_ROOT/stage1/records.jsonl" \
    --source-id synthetic-3d-cartoon-006 \
    --source-id synthetic-3d-cartoon-009 \
    --source-id synthetic-3d-cartoon-010 \
    --source-id synthetic-3d-cartoon-012 \
    --source-id synthetic-3d-cartoon-016 \
    --source-id synthetic-3d-cartoon-022 \
    --source-id synthetic-3d-cartoon-027 \
    --source-id synthetic-3d-cartoon-031 \
    --required-style 3d_cartoon \
    --probe-stage batch \
    --prompt-stage stage2 \
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
  echo "PAIR_BANK_3D_STAGE2_EXIT_CODE=$runner_rc"
fi
```

Expected: `records=8`, `images=8`, `failures=0`.

## Clay representative subset: 12

```bash
set +e
export RUN_ROOT="$FACE_DESTYLE_ROOT/outputs/pair-bank-clay-v1"
export RUN="$RUN_ROOT/stage2-sequential"

if test -e "$RUN"; then
  echo "STOP: output path already exists: $RUN"
else
  mkdir -p "$RUN/images"
  python scripts/run_flux_kontext_probe.py \
    --input-records "$RUN_ROOT/stage1/records.jsonl" \
    --source-id matv2-clay-001 \
    --source-id matv2-clay-002 \
    --source-id matv2-clay-005 \
    --source-id matv2-clay-007 \
    --source-id matv2-clay-009 \
    --source-id matv2-clay-010 \
    --source-id matv2-clay-012 \
    --source-id matv2-clay-013 \
    --source-id matv2-clay-015 \
    --source-id matv2-clay-016 \
    --source-id matv2-clay-021 \
    --source-id matv2-clay-024 \
    --required-style clay \
    --probe-stage batch \
    --prompt-stage stage2 \
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
  echo "PAIR_BANK_CLAY_STAGE2_EXIT_CODE=$runner_rc"
fi
```

Expected: `records=12`, `images=12`, `failures=0`.

## Origami residual-style subset: 10

```bash
set +e
export RUN_ROOT="$FACE_DESTYLE_ROOT/outputs/pair-bank-origami-v1"
export RUN="$RUN_ROOT/stage2-sequential"

if test -e "$RUN"; then
  echo "STOP: output path already exists: $RUN"
else
  mkdir -p "$RUN/images"
  python scripts/run_flux_kontext_probe.py \
    --input-records "$RUN_ROOT/stage1/records.jsonl" \
    --source-id matv2-origami-006 \
    --source-id matv2-origami-008 \
    --source-id matv2-origami-014 \
    --source-id matv2-origami-016 \
    --source-id matv2-origami-020 \
    --source-id matv2-origami-021 \
    --source-id matv2-origami-024 \
    --source-id matv2-origami-025 \
    --source-id matv2-origami-027 \
    --source-id matv2-origami-029 \
    --required-style origami \
    --probe-stage batch \
    --prompt-stage stage2 \
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
  echo "PAIR_BANK_ORIGAMI_STAGE2_EXIT_CODE=$runner_rc"
fi
```

Expected: `records=10`, `images=10`, `failures=0`.

Run one style at a time. After each successful run, download that style's entire
`stage2-sequential` directory. The next local review will compare the original styled source,
Stage 1, and true sequential Stage 2. Closed-teacher generation is a separate later route for the
remaining hard cases.
