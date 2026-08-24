# AutoDL Origami Kontext LoRA — approved 23-pair run

This is the frozen first Origami LoRA dataset. It contains 23 condition/target/instruction triplets
that passed full-frame review. `matv2-origami-004` is excluded after three teacher attempts retained
geometric skin marks. The withdrawn `origami-lora-pairs-v1-20` must not be used.

## Upload

Upload this local file:

```text
/Users/pot/Desktop/origami-lora-pairs-v1-23.zip
```

to exactly:

```text
/root/autodl-tmp/face-destyle/packages/origami-lora-pairs-v1-23.zip
```

Expected SHA-256:

```text
3fc67dea41e114d3b06e625efcbf0874824f22f4e58ac60351adf42b005ac47b
```

## 1. Sync code through the AutoDL route

Run in persistent `tmux`. The academic proxy is used only for GitHub and then removed. Fetching
`origin main` explicitly avoids the earlier multiple-branch fast-forward error.

```bash
set -euo pipefail
conda activate face-destyle

export FACE_DESTYLE_ROOT=/root/autodl-tmp/face-destyle
export REPO="$FACE_DESTYLE_ROOT/code/Face-DeStyle-Pipeline"
export MODEL_DIR="$FACE_DESTYLE_ROOT/models/diffusion/FLUX.1-Kontext-dev"
export PACKAGE="$FACE_DESTYLE_ROOT/packages/origami-lora-pairs-v1-23.zip"
export TRAIN_DATA="$FACE_DESTYLE_ROOT/data/origami-lora-pairs-v1-23"
export TRAINER="$FACE_DESTYLE_ROOT/code/diffusers-kontext-training/examples/dreambooth/train_dreambooth_lora_flux_kontext.py"
export OUTPUT_DIR="$FACE_DESTYLE_ROOT/outputs/origami-destyle-lora-teacher23-r16-steps300"
export HF_HOME="$FACE_DESTYLE_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="$FACE_DESTYLE_ROOT/cache/torch"
export PIP_CACHE_DIR="$FACE_DESTYLE_ROOT/cache/pip"

cd "$REPO"
git status --short --branch
test -z "$(git status --porcelain)" || { echo "STOP: preserve local repository changes before sync"; false; }
source /etc/network_turbo
if ! git fetch origin main; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
  echo "STOP: git fetch failed"
  false
fi
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
git merge --ff-only FETCH_HEAD
git log -1 --oneline
```

The block stops automatically if the status shows local modifications.

## 2. Extract and verify the ready-to-train dataset

This uses a new path and never overwrites the withdrawn 20-pair dataset.

```bash
test -f "$PACKAGE" || { echo "STOP: upload missing: $PACKAGE"; false; }
echo "3fc67dea41e114d3b06e625efcbf0874824f22f4e58ac60351adf42b005ac47b  $PACKAGE" | sha256sum -c -
test ! -e "$TRAIN_DATA" || { echo "STOP: output already exists: $TRAIN_DATA"; false; }
unzip -q "$PACKAGE" -d "$FACE_DESTYLE_ROOT/data"

test "$(find "$TRAIN_DATA/train/condition" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 23
test "$(find "$TRAIN_DATA/train/target" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 23
test "$(wc -l < "$TRAIN_DATA/train/metadata.jsonl")" -eq 23
test -f "$TRAIN_DATA/preview.jpg"
echo "DATASET_OK=23"
```

The ImageFolder columns resolve from `target_file_name` and `condition_file_name` to `target` and
`condition`; `instruction` is the per-example caption column.

## 3. Locate the prepared training environment

Do not reinstall or upgrade Torch, Diffusers, Transformers, Accelerate, or the trainer checkout.
Select the existing `accelerate` whose paired Python imports the complete training stack:

```bash
export ACCELERATE=""
while IFS= read -r candidate; do
  candidate_python="${candidate%/bin/accelerate}/bin/python"
  if "$candidate_python" -c 'import torch, diffusers, transformers, accelerate, datasets, peft, bitsandbytes' 2>/dev/null; then
    export ACCELERATE="$candidate"
    export TRAIN_PYTHON="$candidate_python"
    break
  fi
done < <(find /root/miniconda3/envs "$FACE_DESTYLE_ROOT" -type f -path '*/bin/accelerate' -print 2>/dev/null)

test -n "$ACCELERATE" || { echo "STOP: no prepared Kontext training environment found"; false; }
test -f "$TRAINER" || { echo "STOP: trainer missing: $TRAINER"; false; }
test -f "$MODEL_DIR/model_index.json" || { echo "STOP: model root is wrong: $MODEL_DIR"; false; }

echo "ACCELERATE=$ACCELERATE"
echo "TRAIN_PYTHON=$TRAIN_PYTHON"
"$TRAIN_PYTHON" -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")'
"$TRAIN_PYTHON" "$TRAINER" --help | grep -E -- '--dataset_name|--image_column|--cond_image_column|--caption_column|--cache_latents|--rank'
nvidia-smi
```

## 4. Launch the fresh 300-step run

The run saves checkpoints at steps 100, 200, and 300 so later held-out comparison can choose the
least overfit checkpoint. It does not resume or overwrite the historical eight-pair 3D run.

```bash
test ! -e "$OUTPUT_DIR" || { echo "STOP: training output already exists: $OUTPUT_DIR"; false; }
mkdir -p "$OUTPUT_DIR"

set +e
"$ACCELERATE" launch --mixed_precision bf16 "$TRAINER" \
  --pretrained_model_name_or_path "$MODEL_DIR" \
  --dataset_name "$TRAIN_DATA/train" \
  --image_column target \
  --cond_image_column condition \
  --caption_column instruction \
  --instance_prompt "Convert this folded-paper origami portrait into a natural realistic photograph while preserving the person, pose, expression, clothing, composition, and background." \
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
  --max_train_steps 300 \
  --checkpointing_steps 100 \
  --checkpoints_total_limit 3 \
  --rank 16 \
  --seed 42 \
  2>&1 | tee "$OUTPUT_DIR/train.log"

train_exit=${PIPESTATUS[0]}
set -e
echo "ORIGAMI_LORA_TRAIN_EXIT_CODE=$train_exit"
test "$train_exit" -eq 0
test -f "$OUTPUT_DIR/pytorch_lora_weights.safetensors"
find "$OUTPUT_DIR" -maxdepth 2 -type f -printf '%P\n' | sort
```

Do not judge the adapter from training loss alone. The next operation after a successful exit is a
fixed held-out inference comparison of checkpoints 100/200/300 on the six Origami holdouts.

## Provenance

The canonical archive copy is
`/Users/pot/Documents/大创/实验归档/origami-lora-pairs-v1-23.zip`. The 23 teacher targets and prompt
notes remain in the local experiment archive. The tracked frozen
selection is `data/manifests/multistyle-pair-bank/origami_target_selection_v1.csv`. The ready-to-
train archive was generated by `scripts/build_pair_bank_lora_dataset.py` and visually checked via
its 23-pair `preview.jpg`.
