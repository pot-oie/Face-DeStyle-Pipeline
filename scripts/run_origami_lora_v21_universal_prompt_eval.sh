#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/run_origami_lora_v21_universal_prompt_eval.sh

Run V2 checkpoint 200 on the fixed six Origami holdouts with one universal prompt
copied from the V2.1 training core. The frozen seed-42 inference settings are unchanged.
Outputs remain in the AutoDL output directory and are not archived or packaged.
EOF
  exit 0
fi

FACE_DESTYLE_ROOT=${FACE_DESTYLE_ROOT:-/root/autodl-tmp/face-destyle}
REPO=${REPO:-$FACE_DESTYLE_ROOT/code/Face-DeStyle-Pipeline}
DATA_ROOT=${DATA_ROOT:-$FACE_DESTYLE_ROOT/data/Face-DeStyle-Data}
MODEL_DIR=${MODEL_DIR:-$FACE_DESTYLE_ROOT/models/diffusion/FLUX.1-Kontext-dev}
MODEL_MANIFEST=${MODEL_MANIFEST:-$FACE_DESTYLE_ROOT/models/download-manifests/flux1-kontext-dev-modelscope-master.txt}
MODEL_HASHES=${MODEL_HASHES:-$FACE_DESTYLE_ROOT/models/download-manifests/flux1-kontext-dev-modelscope-master.config-files.sha256}
V2_TRAIN_OUTPUT=${V2_TRAIN_OUTPUT:-$FACE_DESTYLE_ROOT/outputs/origami-destyle-lora-v2-51-r16-steps200}
RUN_ROOT=${RUN_ROOT:-$FACE_DESTYLE_ROOT/outputs/origami-lora-v2-universal-prompt-six-seed42}

SOURCE_LIST=$REPO/data/manifests/multistyle-pair-bank/origami_sources.csv
PROMPT_OVERRIDES=$REPO/configs/eval/origami_v21_universal_holdout6.json
HOLDOUT_LIST=$RUN_ROOT/holdout_sources.csv
WEIGHTS=$V2_TRAIN_OUTPUT/checkpoint-200/pytorch_lora_weights.safetensors
METHOD_DIR=$RUN_ROOT/v2-checkpoint-200-universal-prompt

on_exit() {
  code=$?
  echo "ORIGAMI_V21_UNIVERSAL_PROMPT_EXIT_CODE=$code" | tee "$RUN_ROOT/LAST_EXIT.txt"
}
trap on_exit EXIT

cd "$REPO"
mkdir -p "$RUN_ROOT"
for required in \
  "$SOURCE_LIST" "$PROMPT_OVERRIDES" "$MODEL_DIR/model_index.json" \
  "$MODEL_MANIFEST" "$MODEL_HASHES" "$WEIGHTS"
do
  [[ -f "$required" ]] || { echo "STOP: missing required file: $required"; exit 1; }
done

awk -F, 'BEGIN {OFS=","} NR==1 {print; next} $4=="holdout" {$4="candidate"; print}' \
  "$SOURCE_LIST" > "$HOLDOUT_LIST"
holdout_count=$(awk -F, 'NR>1 {count++} END {print count+0}' "$HOLDOUT_LIST")
[[ "$holdout_count" -eq 6 ]] || { echo "STOP: expected 6 holdouts, found $holdout_count"; exit 1; }

records=0
images=0
failures=0
[[ -f "$METHOD_DIR/records.jsonl" ]] && records=$(wc -l < "$METHOD_DIR/records.jsonl")
[[ -d "$METHOD_DIR/images" ]] && images=$(find "$METHOD_DIR/images" -maxdepth 1 -type f -name '*.png' | wc -l)
[[ -f "$METHOD_DIR/failures.jsonl" ]] && failures=$(wc -l < "$METHOD_DIR/failures.jsonl")

if [[ "$records" -ne 6 || "$images" -ne 6 || "$failures" -ne 0 ]]; then
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
    --lora-weights "$WEIGHTS"
    --lora-scale 1.0
    --output-dir "$METHOD_DIR/images"
    --records-output "$METHOD_DIR/records.jsonl"
    --failures-output "$METHOD_DIR/failures.jsonl"
  )
  if [[ -e "$METHOD_DIR/records.jsonl" || -e "$METHOD_DIR/failures.jsonl" || -d "$METHOD_DIR/images" ]]; then
    args+=(--resume)
  fi
  mkdir -p "$METHOD_DIR"
  "${args[@]}" 2>&1 | tee -a "$METHOD_DIR.log"
fi

records=$(wc -l < "$METHOD_DIR/records.jsonl")
images=$(find "$METHOD_DIR/images" -maxdepth 1 -type f -name '*.png' | wc -l)
failures=0
[[ -f "$METHOD_DIR/failures.jsonl" ]] && failures=$(wc -l < "$METHOD_DIR/failures.jsonl")
echo "OUTPUT_IMAGES=$METHOD_DIR/images"
echo "RECORDS=$records IMAGES=$images FAILURES=$failures"
[[ "$records" -eq 6 && "$images" -eq 6 && "$failures" -eq 0 ]]
