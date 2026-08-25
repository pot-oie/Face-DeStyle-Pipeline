# AutoDL Origami LoRA V2 prompt-alignment probe

This is a three-image diagnostic using V2 checkpoint 200 on the unresolved holdouts `002`,
`011`, and `018`. It keeps seed 42, 28 steps, guidance 2.5, and LoRA scale 1.0 fixed while
replacing the generic Origami evaluation instruction with a source-specific full-subject prompt.
It does not train, resume training, overwrite the six-holdout comparison, or sweep parameters.

Update the AutoDL checkout through Git, then launch the runner inside `screen`:

```bash
set -Eeuo pipefail
export FACE_DESTYLE_ROOT=/root/autodl-tmp/face-destyle
export REPO="$FACE_DESTYLE_ROOT/code/Face-DeStyle-Pipeline"
export PROBE_SCRIPT="$REPO/scripts/run_origami_lora_v2_prompt_alignment_probe.sh"
export PROBE_LOG="$FACE_DESTYLE_ROOT/outputs/origami-lora-v2-prompt-alignment-hard3-screen.log"

cd "$REPO"
source /etc/network_turbo
git fetch origin main
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
git merge --ff-only origin/main
bash "$PROBE_SCRIPT" --help
bash -n "$PROBE_SCRIPT"

screen -dmS origami-prompt-hard3 bash -lc \
  "source /root/miniconda3/etc/profile.d/conda.sh && conda activate face-destyle && cd '$REPO' && bash '$PROBE_SCRIPT' > '$PROBE_LOG' 2>&1"

echo "PROBE_SCREEN=origami-prompt-hard3"
echo "PROBE_LOG=$PROBE_LOG"
screen -ls 2>&1 || true
```

Follow it without attaching:

```bash
tail -f /root/autodl-tmp/face-destyle/outputs/origami-lora-v2-prompt-alignment-hard3-screen.log
```

Successful archive:

```text
/root/autodl-tmp/face-destyle/packages/origami-lora-v2-prompt-alignment-hard3-seed42.zip
```
