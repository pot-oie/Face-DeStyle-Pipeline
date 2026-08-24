# AutoDL Origami LoRA dataset preparation — withdrawn

> **STOP: do not build or train `origami-lora-pairs-v1-20`.**
>
> A face-only review incorrectly accepted 19 open-model targets that still contain obvious
> folded-paper hair, headwear, clothing, or bust material. Under the corrected full-portrait
> standard, five closed-teacher targets (`008`, `021`, `025`, `027`, and `029`) are accepted. Five
> pairs are still not a viable LoRA dataset. The next step is additional closed-teacher generation
> and strict review.

This document is retained to explain the rejected experiment and the AutoDL synchronization
failure. The old 3D LoRA must remain untouched.

## Teacher targets retained for future dataset work

The accepted local files are:

```text
/Users/pot/Documents/大创/实验归档/multistyle-pair-bank-stage1-review-20260824/origami/closed-teacher/images/matv2-origami-008.png
/Users/pot/Documents/大创/实验归档/multistyle-pair-bank-stage1-review-20260824/origami/closed-teacher/images/matv2-origami-021.png
/Users/pot/Documents/大创/实验归档/multistyle-pair-bank-stage1-review-20260824/origami/closed-teacher/images/matv2-origami-025.png
/Users/pot/Documents/大创/实验归档/multistyle-pair-bank-stage1-review-20260824/origami/closed-teacher/images/matv2-origami-027.png
/Users/pot/Documents/大创/实验归档/multistyle-pair-bank-stage1-review-20260824/origami/closed-teacher/images/matv2-origami-029.png
```

Their eventual AutoDL destination is:

```text
/root/autodl-tmp/face-destyle/outputs/pair-bank-origami-v1/closed-teacher/images/
```

Do not combine this target with the 19 rejected Stage 1 and Stage 2 outputs. They are useful as
failure evidence only.

## Sync the corrected repository

Run inside persistent `tmux`:

```bash
conda activate face-destyle

export FACE_DESTYLE_ROOT=/root/autodl-tmp/face-destyle
export REPO="$FACE_DESTYLE_ROOT/code/Face-DeStyle-Pipeline"

cd "$REPO"
git status --short --branch
source /etc/network_turbo
git fetch origin main
git merge --ff-only FETCH_HEAD
git log -1 --oneline
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
```

If `git status --short --branch` shows local modifications, stop before merging and return that
output. `git fetch origin main` followed by `git merge --ff-only FETCH_HEAD` avoids the
`Cannot fast-forward to multiple branches` failure caused by ambiguous pull configuration.

The previous missing-script and missing-directory errors were consequences of the failed Git sync:
the repository never advanced to the commit containing the builder, so the builder did not run and
could not create the dataset tree. They are not separate Python or filesystem failures.

## Historical trainer audit — do not launch training

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

The commands above are safe read-only checks, but there is currently no approved Origami dataset to
train. Do not launch a new adapter, resume a checkpoint, or overwrite the eight-pair 3D checkpoint.
