# AutoDL Origami LoRA V2 fixed holdout evaluation

The V2 training run completed all 200 steps and saved valid LoRA weights at checkpoints 50, 100,
150, and 200 plus the final adapter. Do not retrain or resume it. The next experiment is the
unchanged six-holdout seed-42 comparison across:

```text
Base
frozen V1 checkpoint 100
V2 checkpoint 50
V2 checkpoint 100
V2 checkpoint 150
V2 checkpoint 200
```

The runner uses the same six holdouts, stage-1 Origami instruction, seed 42, 28 inference steps,
guidance 2.5, and LoRA scale 1.0 as the first comparison. Complete method directories are skipped;
an interrupted method resumes through the existing runner.

Upload the local script:

```text
/Users/pot/Github/Face-DeStyle-Pipeline/scripts/run_origami_lora_v2_holdout_eval.sh
```

to exactly:

```text
/root/autodl-tmp/face-destyle/code/Face-DeStyle-Pipeline/scripts/run_origami_lora_v2_holdout_eval.sh
```

Then launch it in `screen`:

```bash
set -u
export FACE_DESTYLE_ROOT=/root/autodl-tmp/face-destyle
export REPO="$FACE_DESTYLE_ROOT/code/Face-DeStyle-Pipeline"
export EVAL_SCRIPT="$REPO/scripts/run_origami_lora_v2_holdout_eval.sh"
export EVAL_LOG="$FACE_DESTYLE_ROOT/outputs/origami-lora-heldout-v2-screen.log"

if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "STOP: upload missing: $EVAL_SCRIPT"
else
  bash "$EVAL_SCRIPT" --help
  bash -n "$EVAL_SCRIPT"
  screen -dmS origami-eval-v2 bash -lc \
    "source /root/miniconda3/etc/profile.d/conda.sh && conda activate face-destyle && cd '$REPO' && bash '$EVAL_SCRIPT' > '$EVAL_LOG' 2>&1"
  echo "EVAL_SCREEN=origami-eval-v2"
  echo "EVAL_LOG=$EVAL_LOG"
  screen -ls 2>&1 || true
fi
```

Follow progress without attaching:

```bash
tail -f /root/autodl-tmp/face-destyle/outputs/origami-lora-heldout-v2-screen.log
```

The successful archive destination is:

```text
/root/autodl-tmp/face-destyle/packages/origami-lora-heldout-v2-base-v1ckpt100-v2ckpt50-100-150-200-seed42.zip
```
