# AutoDL Origami LoRA V2.1 — universal prompt and template-caption run

Do not run either operation until AutoDL is online. Both use `screen`, not `tmux`. Step 1 keeps
V2 checkpoint 200 frozen and writes six loose PNG files without an archive. Step 2 is optional: it
copies the existing 51-pair V2 ImageFolder into a new V2.1 directory, rewrites only metadata to five
concise prompt templates, verifies every template against the model's local 77-token CLIP
tokenizer, and trains a fresh adapter from Base.

## Step 1: one universal prompt on all six holdouts

Update the repository and launch the diagnostic:

```bash
set -Eeuo pipefail
export FACE_DESTYLE_ROOT=/root/autodl-tmp/face-destyle
export REPO="$FACE_DESTYLE_ROOT/code/Face-DeStyle-Pipeline"
export EVAL_SCRIPT="$REPO/scripts/run_origami_lora_v21_universal_prompt_eval.sh"
export EVAL_LOG="$FACE_DESTYLE_ROOT/outputs/origami-lora-v2-universal-prompt-six-screen.log"

cd "$REPO"
source /etc/network_turbo
git fetch origin main
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
git merge --ff-only origin/main
bash "$EVAL_SCRIPT" --help
bash -n "$EVAL_SCRIPT"

screen -dmS origami-v21-universal bash -lc \
  "source /root/miniconda3/etc/profile.d/conda.sh && conda activate face-destyle && cd '$REPO' && bash '$EVAL_SCRIPT' > '$EVAL_LOG' 2>&1"

echo "EVAL_SCREEN=origami-v21-universal"
echo "EVAL_LOG=$EVAL_LOG"
screen -ls 2>&1 || true
```

Monitor without attaching:

```bash
tail -f /root/autodl-tmp/face-destyle/outputs/origami-lora-v2-universal-prompt-six-screen.log
```

The six uncompressed outputs remain at:

```text
/root/autodl-tmp/face-destyle/outputs/origami-lora-v2-universal-prompt-six-seed42/v2-checkpoint-200-universal-prompt/images
```

Completion must report `RECORDS=6 IMAGES=6 FAILURES=0` and
`ORIGAMI_V21_UNIVERSAL_PROMPT_EXIT_CODE=0`.

## Step 2: optional V2.1 caption-only fresh training

Run this only if Step 1 is not good enough. It does not use or resume any old adapter. The builder
verifies that the new dataset has 51 pairs, excludes all protected IDs, and uses template counts
`11/10/10/10/10`. The adjusted `clip77` paths intentionally differ from the first V2.1 draft, so
an earlier metadata directory cannot be trained accidentally.

```bash
set -Eeuo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate face-destyle

export FACE_DESTYLE_ROOT=/root/autodl-tmp/face-destyle
export REPO="$FACE_DESTYLE_ROOT/code/Face-DeStyle-Pipeline"
cd "$REPO"

bash scripts/launch_origami_lora_v21_train.sh --help
bash -n scripts/launch_origami_lora_v21_train.sh
bash scripts/launch_origami_lora_v21_train.sh
```

The launcher creates and verifies:

```text
/root/autodl-tmp/face-destyle/data/origami-lora-pairs-v21-51-clip77
```

It then starts `screen` session `origami-lora-v21-clip77` with fresh Base-model training at rank 16,
learning rate `1e-4`, effective batch 4, and at most 200 steps. It saves checkpoints 50, 100, 150,
and 200 under:

```text
/root/autodl-tmp/face-destyle/outputs/origami-destyle-lora-v21-51-clip77-r16-steps200
```

Monitor without attaching:

```bash
tail -f /root/autodl-tmp/face-destyle/outputs/origami-destyle-lora-v21-51-clip77-r16-steps200/screen.log
```

Or attach and later detach with `Ctrl-A`, then `D`:

```bash
screen -r origami-lora-v21-clip77
```
