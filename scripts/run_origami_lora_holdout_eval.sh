#!/usr/bin/env bash
set -Eeuo pipefail

FACE_DESTYLE_ROOT=${FACE_DESTYLE_ROOT:-/root/autodl-tmp/face-destyle}
REPO=${REPO:-$FACE_DESTYLE_ROOT/code/Face-DeStyle-Pipeline}
DATA_ROOT=${DATA_ROOT:-$FACE_DESTYLE_ROOT/data/Face-DeStyle-Data}
MODEL_DIR=${MODEL_DIR:-$FACE_DESTYLE_ROOT/models/diffusion/FLUX.1-Kontext-dev}
MODEL_MANIFEST=${MODEL_MANIFEST:-$FACE_DESTYLE_ROOT/models/download-manifests/flux1-kontext-dev-modelscope-master.txt}
MODEL_HASHES=${MODEL_HASHES:-$FACE_DESTYLE_ROOT/models/download-manifests/flux1-kontext-dev-modelscope-master.config-files.sha256}
TRAIN_OUTPUT=${TRAIN_OUTPUT:-$FACE_DESTYLE_ROOT/outputs/origami-destyle-lora-teacher23-r16-steps300}
RUN_ROOT=${RUN_ROOT:-$FACE_DESTYLE_ROOT/outputs/origami-lora-heldout-base-ckpt100-200-300-seed42}
ARCHIVE=${ARCHIVE:-$FACE_DESTYLE_ROOT/packages/origami-lora-heldout-base-ckpt100-200-300-seed42.zip}

SOURCE_LIST=$REPO/data/manifests/multistyle-pair-bank/origami_sources.csv
HOLDOUT_LIST=$RUN_ROOT/holdout_sources.csv

on_exit() {
  code=$?
  echo "ORIGAMI_HOLDOUT_EVAL_EXIT_CODE=$code"
  if [[ $code -eq 0 ]]; then
    echo "ARCHIVE=$ARCHIVE"
  fi
}
trap on_exit EXIT

cd "$REPO"

for required in \
  "$SOURCE_LIST" \
  "$MODEL_DIR/model_index.json" \
  "$MODEL_MANIFEST" \
  "$MODEL_HASHES" \
  "$TRAIN_OUTPUT/checkpoint-100/pytorch_lora_weights.safetensors" \
  "$TRAIN_OUTPUT/checkpoint-200/pytorch_lora_weights.safetensors" \
  "$TRAIN_OUTPUT/checkpoint-300/pytorch_lora_weights.safetensors"
do
  [[ -f "$required" ]] || { echo "STOP: missing required file: $required"; exit 1; }
done

[[ ! -e "$ARCHIVE" ]] || { echo "STOP: archive already exists: $ARCHIVE"; exit 1; }

mkdir -p "$RUN_ROOT/source-images" "$(dirname "$ARCHIVE")"

awk -F, 'BEGIN {OFS=","} NR==1 {print; next} $4=="holdout" {$4="candidate"; print}' \
  "$SOURCE_LIST" > "$HOLDOUT_LIST"

holdout_count=$(awk -F, 'NR>1 {count++} END {print count+0}' "$HOLDOUT_LIST")
[[ "$holdout_count" -eq 6 ]] || { echo "STOP: expected 6 holdouts, found $holdout_count"; exit 1; }

while IFS=, read -r source_id asset_path _style _role _notes; do
  [[ "$source_id" == "source_id" ]] && continue
  cp "$DATA_ROOT/$asset_path" "$RUN_ROOT/source-images/$source_id.png"
done < "$HOLDOUT_LIST"

run_method() {
  label=$1
  weights=${2:-}
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
    --probe-stage batch
    --model-dir "$MODEL_DIR"
    --download-manifest "$MODEL_MANIFEST"
    --hash-manifest "$MODEL_HASHES"
    --styles-config configs/styles.yaml
    --seed 42
    --num-inference-steps 28
    --guidance-scale 2.5
    --output-dir "$method_dir/images"
    --records-output "$method_dir/records.jsonl"
    --failures-output "$method_dir/failures.jsonl"
  )
  if [[ -n "$weights" ]]; then
    args+=(--lora-weights "$weights" --lora-scale 1.0)
  fi
  if [[ -e "$method_dir/records.jsonl" || -e "$method_dir/failures.jsonl" || -d "$method_dir/images" ]]; then
    args+=(--resume)
  fi
  echo "START_METHOD=$label"
  "${args[@]}" 2>&1 | tee -a "$method_dir.log"
  records=$(wc -l < "$method_dir/records.jsonl")
  images=$(find "$method_dir/images" -maxdepth 1 -type f -name '*.png' | wc -l)
  failures=0
  [[ -f "$method_dir/failures.jsonl" ]] && failures=$(wc -l < "$method_dir/failures.jsonl")
  echo "METHOD=$label RECORDS=$records IMAGES=$images FAILURES=$failures"
  [[ "$records" -eq 6 && "$images" -eq 6 && "$failures" -eq 0 ]]
}

run_method base
run_method checkpoint-100 "$TRAIN_OUTPUT/checkpoint-100/pytorch_lora_weights.safetensors"
run_method checkpoint-200 "$TRAIN_OUTPUT/checkpoint-200/pytorch_lora_weights.safetensors"
run_method checkpoint-300 "$TRAIN_OUTPUT/checkpoint-300/pytorch_lora_weights.safetensors"

{
  echo "schema=origami-lora-heldout-comparison/v1"
  echo "seed=42"
  echo "num_inference_steps=28"
  echo "guidance_scale=2.5"
  echo "lora_scale=1.0"
  echo "holdouts=6"
  echo "methods=base,checkpoint-100,checkpoint-200,checkpoint-300"
  sha256sum \
    "$TRAIN_OUTPUT/checkpoint-100/pytorch_lora_weights.safetensors" \
    "$TRAIN_OUTPUT/checkpoint-200/pytorch_lora_weights.safetensors" \
    "$TRAIN_OUTPUT/checkpoint-300/pytorch_lora_weights.safetensors"
} > "$RUN_ROOT/RUN_SUMMARY.txt"

archive_parent=$(dirname "$RUN_ROOT")
archive_name=$(basename "$RUN_ROOT")
(
  cd "$archive_parent"
  zip -qr "$ARCHIVE" "$archive_name"
)

echo "ARCHIVE_SHA256=$(sha256sum "$ARCHIVE" | awk '{print $1}')"
du -sh "$RUN_ROOT" "$ARCHIVE"
