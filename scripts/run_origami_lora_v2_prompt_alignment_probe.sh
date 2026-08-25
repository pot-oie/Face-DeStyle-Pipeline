#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/run_origami_lora_v2_prompt_alignment_probe.sh

Run a source-specific prompt-alignment probe on Origami holdouts 002, 011, and 018
with V2 checkpoint 200. The frozen seed-42 inference settings are unchanged. Existing
complete output is skipped; an interrupted run resumes. Environment variables may
override the default AutoDL paths.
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
RUN_ROOT=${RUN_ROOT:-$FACE_DESTYLE_ROOT/outputs/origami-lora-v2-prompt-alignment-hard3-seed42}
ARCHIVE=${ARCHIVE:-$FACE_DESTYLE_ROOT/packages/origami-lora-v2-prompt-alignment-hard3-seed42.zip}

SOURCE_LIST=$REPO/data/manifests/multistyle-pair-bank/origami_sources.csv
PROMPT_OVERRIDES=$REPO/configs/eval/origami_v2_prompt_alignment_hard3.json
PROBE_LIST=$RUN_ROOT/probe_sources.csv
WEIGHTS=$V2_TRAIN_OUTPUT/checkpoint-200/pytorch_lora_weights.safetensors
METHOD_DIR=$RUN_ROOT/v2-checkpoint-200-matched-prompts

on_exit() {
  code=$?
  echo "ORIGAMI_V2_PROMPT_ALIGNMENT_EXIT_CODE=$code" | tee "$RUN_ROOT/LAST_EXIT.txt"
  if [[ $code -eq 0 ]]; then
    echo "ARCHIVE=$ARCHIVE"
  fi
}
trap on_exit EXIT

cd "$REPO"
mkdir -p "$RUN_ROOT" "$RUN_ROOT/source-images" "$(dirname "$ARCHIVE")"

required_files=(
  "$SOURCE_LIST"
  "$PROMPT_OVERRIDES"
  "$MODEL_DIR/model_index.json"
  "$MODEL_MANIFEST"
  "$MODEL_HASHES"
  "$WEIGHTS"
)
for required in "${required_files[@]}"; do
  [[ -f "$required" ]] || { echo "STOP: missing required file: $required"; exit 1; }
done

[[ ! -e "$ARCHIVE" ]] || { echo "STOP: archive already exists: $ARCHIVE"; exit 1; }

awk -F, 'BEGIN {OFS=","} NR==1 {print; next} $1=="matv2-origami-002" || $1=="matv2-origami-011" || $1=="matv2-origami-018" {$4="candidate"; print}' \
  "$SOURCE_LIST" > "$PROBE_LIST"

probe_count=$(awk -F, 'NR>1 {count++} END {print count+0}' "$PROBE_LIST")
[[ "$probe_count" -eq 3 ]] || { echo "STOP: expected 3 probe sources, found $probe_count"; exit 1; }

while IFS=, read -r source_id asset_path _style _role _notes; do
  [[ "$source_id" == "source_id" ]] && continue
  cp "$DATA_ROOT/$asset_path" "$RUN_ROOT/source-images/$source_id.png"
done < "$PROBE_LIST"

records=0
images=0
failures=0
[[ -f "$METHOD_DIR/records.jsonl" ]] && records=$(wc -l < "$METHOD_DIR/records.jsonl")
[[ -d "$METHOD_DIR/images" ]] && images=$(find "$METHOD_DIR/images" -maxdepth 1 -type f -name '*.png' | wc -l)
[[ -f "$METHOD_DIR/failures.jsonl" ]] && failures=$(wc -l < "$METHOD_DIR/failures.jsonl")

if [[ "$records" -ne 3 || "$images" -ne 3 || "$failures" -ne 0 ]]; then
  args=(
    python scripts/run_flux_kontext_probe.py
    --source-list "$PROBE_LIST"
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
echo "RECORDS=$records IMAGES=$images FAILURES=$failures"
[[ "$records" -eq 3 && "$images" -eq 3 && "$failures" -eq 0 ]]

{
  echo "schema=origami-lora-v2-prompt-alignment/v1"
  echo "source_ids=matv2-origami-002,matv2-origami-011,matv2-origami-018"
  echo "adapter=v2-checkpoint-200"
  echo "seed=42"
  echo "num_inference_steps=28"
  echo "guidance_scale=2.5"
  echo "lora_scale=1.0"
  sha256sum "$PROMPT_OVERRIDES" "$WEIGHTS"
} > "$RUN_ROOT/RUN_SUMMARY.txt"

archive_parent=$(dirname "$RUN_ROOT")
archive_name=$(basename "$RUN_ROOT")
(
  cd "$archive_parent"
  zip -qr "$ARCHIVE" "$archive_name"
)

echo "ARCHIVE_SHA256=$(sha256sum "$ARCHIVE" | awk '{print $1}')"
du -sh "$RUN_ROOT" "$ARCHIVE"
