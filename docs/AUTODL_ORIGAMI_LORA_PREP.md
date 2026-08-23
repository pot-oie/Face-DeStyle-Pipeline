# AutoDL Origami LoRA dataset preparation

The reviewed Origami bank now contains 20 accepted source-target pairs: 14 Stage 1 targets, five
true sequential Stage 2 targets, and one closed-teacher target. This document prepares the dataset
and audits the already-installed training environment. It does not train yet and does not touch the
old 3D LoRA.

## Upload the one local teacher target

Upload this local file:

```text
/Users/pot/Documents/大创/实验归档/multistyle-pair-bank-stage1-review-20260824/origami/closed-teacher/images/matv2-origami-008.png
```

to exactly:

```text
/root/autodl-tmp/face-destyle/outputs/pair-bank-origami-v1/closed-teacher/images/matv2-origami-008.png
```

The other 19 selected targets are already in the existing Stage 1 and Stage 2 output directories.

## Sync and build the 20-pair dataset

Run inside persistent `tmux`:

```bash
conda activate face-destyle

export FACE_DESTYLE_ROOT=/root/autodl-tmp/face-destyle
export REPO="$FACE_DESTYLE_ROOT/code/Face-DeStyle-Pipeline"
export DATA_ROOT="$FACE_DESTYLE_ROOT/data/Face-DeStyle-Data"
export RUN_DIR="$FACE_DESTYLE_ROOT/outputs/pair-bank-origami-v1"
export TRAIN_DATA="$FACE_DESTYLE_ROOT/data/origami-lora-pairs-v1-20"

cd "$REPO"
git status --short --branch
source /etc/network_turbo
git pull --ff-only origin main
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

test -f "$RUN_DIR/closed-teacher/images/matv2-origami-008.png"
test ! -e "$TRAIN_DATA"

python scripts/build_pair_bank_lora_dataset.py \
  --source-list data/manifests/multistyle-pair-bank/origami_sources.csv \
  --selection data/manifests/multistyle-pair-bank/origami_target_selection_v1.csv \
  --data-root "$DATA_ROOT" \
  --run-dir "$RUN_DIR" \
  --output-dir "$TRAIN_DATA" \
  --styles-config configs/styles.yaml \
  --style-category origami

find "$TRAIN_DATA/train/condition" -maxdepth 1 -type f -name '*.png' | wc -l
find "$TRAIN_DATA/train/target" -maxdepth 1 -type f -name '*.png' | wc -l
wc -l "$TRAIN_DATA/train/metadata.jsonl"
```

All three counts must be 20. The builder refuses an existing output directory rather than merging
or overwriting it.

## Audit the prepared trainer environment

The earlier 3D run proved that a compatible trainer existed, but a later shell could not find
`accelerate`. Do not reinstall or upgrade Torch, Diffusers, Transformers, or Accelerate. Locate the
prepared environment first:

```bash
conda env list

find /root/miniconda3/envs /root/autodl-tmp/face-destyle \
  -type f -path '*/bin/accelerate' -print 2>/dev/null

export TRAINER="$FACE_DESTYLE_ROOT/code/diffusers-kontext-training/examples/dreambooth/train_dreambooth_lora_flux_kontext.py"
test -f "$TRAINER" && echo "trainer=ok" || echo "STOP: trainer missing"

python "$TRAINER" --help | sed -n '1,220p'
```

Return the `conda env list`, discovered `accelerate` path, and trainer help output before launching
training. The intended next run is a new Origami-specific adapter with a separate output directory;
it must not resume or overwrite the eight-pair 3D checkpoint.
