#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/launch_origami_lora_v21_train.sh

Build an independent 51-pair V2.1 ImageFolder dataset from the existing V2 dataset by
rewriting only its instructions to five concise templates, verify it, and launch a fresh
Base-model rank-16/200-step LoRA training run in screen. Never resumes V1 or V2 weights.
EOF
  exit 0
fi

FACE_DESTYLE_ROOT=${FACE_DESTYLE_ROOT:-/root/autodl-tmp/face-destyle}
REPO=${REPO:-$FACE_DESTYLE_ROOT/code/Face-DeStyle-Pipeline}
SOURCE_DATA=${SOURCE_DATA:-$FACE_DESTYLE_ROOT/data/origami-lora-pairs-v2-51}
TRAIN_DATA=${TRAIN_DATA:-$FACE_DESTYLE_ROOT/data/origami-lora-pairs-v21-51-clip77}
MODEL_DIR=${MODEL_DIR:-$FACE_DESTYLE_ROOT/models/diffusion/FLUX.1-Kontext-dev}
TRAINER=${TRAINER:-$FACE_DESTYLE_ROOT/code/diffusers-kontext-training/examples/dreambooth/train_dreambooth_lora_flux_kontext.py}
OUTPUT_DIR=${OUTPUT_DIR:-$FACE_DESTYLE_ROOT/outputs/origami-destyle-lora-v21-51-clip77-r16-steps200}
SCREEN_NAME=${SCREEN_NAME:-origami-lora-v21-clip77}
HF_HOME=${HF_HOME:-$FACE_DESTYLE_ROOT/cache/huggingface}
HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}
TORCH_HOME=${TORCH_HOME:-$FACE_DESTYLE_ROOT/cache/torch}
PIP_CACHE_DIR=${PIP_CACHE_DIR:-$FACE_DESTYLE_ROOT/cache/pip}
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TORCH_HOME PIP_CACHE_DIR

cd "$REPO"
test -f "$SOURCE_DATA/train/metadata.jsonl" || { echo "STOP: V2 source dataset missing: $SOURCE_DATA"; exit 1; }
test -f "$MODEL_DIR/model_index.json" || { echo "STOP: model missing: $MODEL_DIR"; exit 1; }
test -f "$TRAINER" || { echo "STOP: trainer missing: $TRAINER"; exit 1; }
test ! -e "$OUTPUT_DIR" || { echo "STOP: fresh output path already exists: $OUTPUT_DIR"; exit 1; }

export ACCELERATE=""
while IFS= read -r candidate; do
  candidate_python="${candidate%/bin/accelerate}/bin/python"
  if "$candidate_python" -c \
    'import torch, diffusers, transformers, accelerate, datasets, peft, bitsandbytes' \
    2>/dev/null
  then
    export ACCELERATE="$candidate"
    export TRAIN_PYTHON="$candidate_python"
    break
  fi
done < <(find /root/miniconda3/envs "$FACE_DESTYLE_ROOT" -type f -path '*/bin/accelerate' -print 2>/dev/null)

test -n "$ACCELERATE" || { echo "STOP: no prepared Kontext training environment found"; exit 1; }
"$TRAIN_PYTHON" -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"); assert torch.cuda.is_available()'

if [[ -e "$TRAIN_DATA" ]]; then
  "$TRAIN_PYTHON" scripts/build_origami_lora_v21_dataset.py \
    --output "$TRAIN_DATA" \
    --clip-tokenizer "$MODEL_DIR/tokenizer" \
    --verify-only
else
  "$TRAIN_PYTHON" scripts/build_origami_lora_v21_dataset.py \
    --source "$SOURCE_DATA" \
    --output "$TRAIN_DATA" \
    --clip-tokenizer "$MODEL_DIR/tokenizer"
fi

if screen -list | grep -q "[.]$SCREEN_NAME"; then
  echo "STOP: screen session already exists: $SCREEN_NAME"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
screen -L -Logfile "$OUTPUT_DIR/screen.log" -dmS "$SCREEN_NAME" \
  "$ACCELERATE" launch --num_processes 1 --mixed_precision bf16 "$TRAINER" \
  --pretrained_model_name_or_path "$MODEL_DIR" \
  --dataset_name "$TRAIN_DATA/train" \
  --image_column target \
  --cond_image_column condition \
  --caption_column instruction \
  --instance_prompt "Make the entire origami portrait a natural photo. Remove paper from visible skin, scalp, hair, headwear, neck, clothes, shoulders, bust, pedestal and support. Keep identity, age, skin tone, facial hair, pose, gaze, expression, composition, background and lighting." \
  --output_dir "$OUTPUT_DIR" \
  --mixed_precision bf16 \
  --resolution 1024 \
  --train_batch_size 1 \
  --guidance_scale 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --optimizer adamw \
  --use_8bit_adam \
  --cache_latents \
  --learning_rate 1e-4 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --max_train_steps 200 \
  --checkpointing_steps 50 \
  --checkpoints_total_limit 4 \
  --rank 16 \
  --seed 42

echo "TRAIN_SCREEN=$SCREEN_NAME"
echo "TRAIN_LOG=$OUTPUT_DIR/screen.log"
echo "TRAIN_OUTPUT=$OUTPUT_DIR"
screen -ls
