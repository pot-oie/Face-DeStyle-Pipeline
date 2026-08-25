# AutoDL Origami Kontext LoRA V2 — 51-pair fresh run

Do not execute this handoff until the operator explicitly reports that AutoDL is online. This V2
run starts from the base `FLUX.1-Kontext-dev` model; it never resumes the selected V1 checkpoint
100. Use `screen`, not `tmux`.

## Upload

Upload the local package:

```text
/Users/pot/Documents/大创/实验归档/origami-lora-pairs-v2-51.zip
```

to exactly:

```text
/root/autodl-tmp/face-destyle/packages/origami-lora-pairs-v2-51.zip
```

The archive expands to
`/root/autodl-tmp/face-destyle/data/origami-lora-pairs-v2-51`. It contains the unchanged 23 strict
V1 pairs plus the independently retained 28 hard V2 pairs. The six holdouts and hard V2 IDs `021`
and `023` are excluded.

## Directly copyable preparation and training block

Run this only after the upload is complete and a GPU is allocated:

```bash
set -Eeuo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate face-destyle

export FACE_DESTYLE_ROOT=/root/autodl-tmp/face-destyle
export PACKAGE="$FACE_DESTYLE_ROOT/packages/origami-lora-pairs-v2-51.zip"
export TRAIN_DATA="$FACE_DESTYLE_ROOT/data/origami-lora-pairs-v2-51"
export MODEL_DIR="$FACE_DESTYLE_ROOT/models/diffusion/FLUX.1-Kontext-dev"
export TRAINER="$FACE_DESTYLE_ROOT/code/diffusers-kontext-training/examples/dreambooth/train_dreambooth_lora_flux_kontext.py"
export OUTPUT_DIR="$FACE_DESTYLE_ROOT/outputs/origami-destyle-lora-v2-51-r16-steps200"
export HF_HOME="$FACE_DESTYLE_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="$FACE_DESTYLE_ROOT/cache/torch"
export PIP_CACHE_DIR="$FACE_DESTYLE_ROOT/cache/pip"

nvidia-smi
test -f "$PACKAGE" || { echo "STOP: upload missing: $PACKAGE"; false; }
test -f "$MODEL_DIR/model_index.json" || { echo "STOP: model missing: $MODEL_DIR"; false; }
test -f "$TRAINER" || { echo "STOP: trainer missing: $TRAINER"; false; }
test ! -e "$TRAIN_DATA" || { echo "STOP: dataset path already exists: $TRAIN_DATA"; false; }
test ! -e "$OUTPUT_DIR" || { echo "STOP: fresh output path already exists: $OUTPUT_DIR"; false; }

unzip -q "$PACKAGE" -d "$FACE_DESTYLE_ROOT/data"
test "$(find "$TRAIN_DATA/train/condition" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 51
test "$(find "$TRAIN_DATA/train/target" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 51
test "$(wc -l < "$TRAIN_DATA/train/metadata.jsonl")" -eq 51
for source_id in \
  matv2-origami-002 matv2-origami-007 matv2-origami-011 \
  matv2-origami-018 matv2-origami-023 matv2-origami-030 \
  origami-hard-v2-021 origami-hard-v2-023
do
  test ! -e "$TRAIN_DATA/train/condition/$source_id.png"
done

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

test -n "$ACCELERATE" || { echo "STOP: no prepared Kontext training environment found"; false; }
"$TRAIN_PYTHON" -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"); assert torch.cuda.is_available()'
if screen -list | grep -q '[.]origami-lora-v2-51'; then
  echo "STOP: screen session origami-lora-v2-51 already exists"
  false
fi

mkdir -p "$OUTPUT_DIR"
screen -L -Logfile "$OUTPUT_DIR/screen.log" -dmS origami-lora-v2-51 \
  "$ACCELERATE" launch --num_processes 1 --mixed_precision bf16 "$TRAINER" \
  --pretrained_model_name_or_path "$MODEL_DIR" \
  --dataset_name "$TRAIN_DATA/train" \
  --image_column target \
  --cond_image_column condition \
  --caption_column instruction \
  --instance_prompt "Convert the complete folded-paper origami subject into a natural photorealistic camera portrait while preserving identity and composition." \
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

echo "TRAIN_SCREEN=origami-lora-v2-51"
echo "TRAIN_LOG=$OUTPUT_DIR/screen.log"
screen -ls
```

Attach without terminating the job with:

```bash
screen -r origami-lora-v2-51
```

Detach with `Ctrl-A`, then `D`. A successful run must leave checkpoints 50, 100, 150, and 200 plus
the final LoRA weights. Do not judge the adapter from training loss; the next operation is the
unchanged six-holdout seed-42 comparison against Base, frozen V1 checkpoint 100, and all four V2
checkpoints.
