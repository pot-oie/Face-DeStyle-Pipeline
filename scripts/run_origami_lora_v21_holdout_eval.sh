#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/run_origami_lora_v21_holdout_eval.sh

Verify the complete V2.1 training run, then generate the fixed six Origami holdouts
for V2.1 checkpoints 50, 100, 150, and 200 with the shared CLIP-safe core prompt.
Outputs remain as loose PNG files; no archive is created. Interrupted methods resume.
EOF
  exit 0
fi

FACE_DESTYLE_ROOT=${FACE_DESTYLE_ROOT:-/root/autodl-tmp/face-destyle}
REPO=${REPO:-$FACE_DESTYLE_ROOT/code/Face-DeStyle-Pipeline}
DATA_ROOT=${DATA_ROOT:-$FACE_DESTYLE_ROOT/data/Face-DeStyle-Data}
MODEL_DIR=${MODEL_DIR:-$FACE_DESTYLE_ROOT/models/diffusion/FLUX.1-Kontext-dev}
MODEL_MANIFEST=${MODEL_MANIFEST:-$FACE_DESTYLE_ROOT/models/download-manifests/flux1-kontext-dev-modelscope-master.txt}
MODEL_HASHES=${MODEL_HASHES:-$FACE_DESTYLE_ROOT/models/download-manifests/flux1-kontext-dev-modelscope-master.config-files.sha256}
V21_TRAIN_OUTPUT=${V21_TRAIN_OUTPUT:-$FACE_DESTYLE_ROOT/outputs/origami-destyle-lora-v21-51-clip77-r16-steps200}
RUN_ROOT=${RUN_ROOT:-$FACE_DESTYLE_ROOT/outputs/origami-lora-v21-heldout-checkpoints50-100-150-200-clip77-seed42}

SOURCE_LIST=$REPO/data/manifests/multistyle-pair-bank/origami_sources.csv
PROMPT_OVERRIDES=$REPO/configs/eval/origami_v21_clip77_universal_holdout6.json
HOLDOUT_LIST=$RUN_ROOT/holdout_sources.csv

on_exit() {
  code=$?
  echo "ORIGAMI_V21_HOLDOUT_EVAL_EXIT_CODE=$code" | tee "$RUN_ROOT/LAST_EXIT.txt"
}
trap on_exit EXIT

cd "$REPO"
mkdir -p "$RUN_ROOT" "$RUN_ROOT/source-images"

python scripts/verify_origami_lora_run.py \
  --output-dir "$V21_TRAIN_OUTPUT" \
  --checkpoint 50 --checkpoint 100 --checkpoint 150 --checkpoint 200 \
  --log "$V21_TRAIN_OUTPUT/screen.log" | tee "$RUN_ROOT/model-verification.log"

for required in \
  "$SOURCE_LIST" "$PROMPT_OVERRIDES" "$MODEL_DIR/model_index.json" \
  "$MODEL_MANIFEST" "$MODEL_HASHES"
do
  [[ -f "$required" ]] || { echo "STOP: missing required file: $required"; exit 1; }
done

awk -F, 'BEGIN {OFS=","} NR==1 {print; next} $4=="holdout" {$4="candidate"; print}' \
  "$SOURCE_LIST" > "$HOLDOUT_LIST"
holdout_count=$(awk -F, 'NR>1 {count++} END {print count+0}' "$HOLDOUT_LIST")
[[ "$holdout_count" -eq 6 ]] || { echo "STOP: expected 6 holdouts, found $holdout_count"; exit 1; }

while IFS=, read -r source_id asset_path _style _role _notes; do
  [[ "$source_id" == "source_id" ]] && continue
  cp "$DATA_ROOT/$asset_path" "$RUN_ROOT/source-images/$source_id.png"
done < "$HOLDOUT_LIST"

run_checkpoint() {
  checkpoint=$1
  label="v21-checkpoint-$checkpoint"
  weights=$V21_TRAIN_OUTPUT/checkpoint-$checkpoint/pytorch_lora_weights.safetensors
  method_dir=$RUN_ROOT/$label
  records=0
  images=0
  failures=0
  [[ -f "$method_dir/records.jsonl" ]] && records=$(wc -l < "$method_dir/records.jsonl")
  [[ -d "$method_dir/images" ]] && images=$(find "$method_dir/images" -maxdepth 1 -type f -name '*.png' | wc -l)
  [[ -f "$method_dir/failures.jsonl" ]] && failures=$(wc -l < "$method_dir/failures.jsonl")
  if [[ "$records" -eq 6 && "$images" -eq 6 && "$failures" -eq 0 ]]; then
    echo "SKIP_METHOD=$label RECORDS=6 IMAGES=6 FAILURES=0"
    return 0
  fi

  args=(
    python scripts/run_flux_kontext_probe.py
    --source-list "$HOLDOUT_LIST"
    --data-root "$DATA_ROOT"
    --required-style origami
    --prompt-stage stage1
    --prompt-overrides "$PROMPT_OVERRIDES"
    --probe-stage batch
    --model-dir "$MODEL_DIR"
    --download-manifest "$MODEL_MANIFEST"
    --hash-manifest "$MODEL_HASHES"
    --styles-config configs/styles.yaml
    --seed 42
    --num-inference-steps 28
    --guidance-scale 2.5
    --lora-weights "$weights"
    --lora-scale 1.0
    --output-dir "$method_dir/images"
    --records-output "$method_dir/records.jsonl"
    --failures-output "$method_dir/failures.jsonl"
  )
  if [[ -e "$method_dir/records.jsonl" || -e "$method_dir/failures.jsonl" || -d "$method_dir/images" ]]; then
    args+=(--resume)
  fi
  mkdir -p "$method_dir"
  echo "START_METHOD=$label"
  "${args[@]}" 2>&1 | tee -a "$method_dir.log"
  records=$(wc -l < "$method_dir/records.jsonl")
  images=$(find "$method_dir/images" -maxdepth 1 -type f -name '*.png' | wc -l)
  failures=0
  [[ -f "$method_dir/failures.jsonl" ]] && failures=$(wc -l < "$method_dir/failures.jsonl")
  echo "METHOD=$label RECORDS=$records IMAGES=$images FAILURES=$failures"
  [[ "$records" -eq 6 && "$images" -eq 6 && "$failures" -eq 0 ]]
}

for checkpoint in 50 100 150 200; do
  run_checkpoint "$checkpoint"
done

{
  echo "schema=origami-lora-v21-holdout/v1"
  echo "seed=42"
  echo "num_inference_steps=28"
  echo "guidance_scale=2.5"
  echo "lora_scale=1.0"
  echo "prompt=clip77-universal-core"
  echo "methods=v21-checkpoint-50,v21-checkpoint-100,v21-checkpoint-150,v21-checkpoint-200"
  sha256sum \
    "$V21_TRAIN_OUTPUT/checkpoint-50/pytorch_lora_weights.safetensors" \
    "$V21_TRAIN_OUTPUT/checkpoint-100/pytorch_lora_weights.safetensors" \
    "$V21_TRAIN_OUTPUT/checkpoint-150/pytorch_lora_weights.safetensors" \
    "$V21_TRAIN_OUTPUT/checkpoint-200/pytorch_lora_weights.safetensors"
} > "$RUN_ROOT/RUN_SUMMARY.txt"

echo "OUTPUT_ROOT=$RUN_ROOT"
echo "METHODS=4 HOLDOUTS_PER_METHOD=6 ARCHIVE=NONE"
