# Non-Origami multistyle routing validation — completed run record

## Scope

Origami is closed: keep V1 checkpoint 100 as a limited approximately 3/6 adapter and do not rerun
V2/V2.1 or the residual hard-three diagnostic. The larger Base FLUX processing validation for
Comic, Ink, Watercolor, 3D cartoon, Clay, and Needle-felt completed on 2026-08-27.

The frozen manifest is
`data/manifests/multistyle-routing/non_origami_validation_137.jsonl`. It contains 137 accepted
sources: 24 each for Comic, Ink, Watercolor, 3D cartoon, and Clay, plus all 17 accepted available
Needle-felt sources. Rejected inputs are not used to balance the count.

This was inference and visual validation only. Both stages completed 137/137 with zero recorded
generation failures, and all Stage 2 inputs matched their same-ID Stage 1 parents. The completed
visual review is in
[`results/multistyle_routing_validation_137_review_20260827.csv`](results/multistyle_routing_validation_137_review_20260827.csv).
Do not rerun it as an active task.

## Historical AutoDL run

Use the existing 32 GB inference instance and environment. No model download or reinstall is
required. Pull the preparation commit before starting.

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate face-destyle

export FACE_DESTYLE_ROOT=/root/autodl-tmp/face-destyle
export REPO="$FACE_DESTYLE_ROOT/code/Face-DeStyle-Pipeline"
export DATA_ROOT="$FACE_DESTYLE_ROOT/data/Face-DeStyle-Data"
export MODEL_DIR="$FACE_DESTYLE_ROOT/models/diffusion/FLUX.1-Kontext-dev"
export MODEL_MANIFEST="$FACE_DESTYLE_ROOT/models/download-manifests/flux1-kontext-dev-modelscope-master.txt"
export MODEL_HASHES="$FACE_DESTYLE_ROOT/models/download-manifests/flux1-kontext-dev-modelscope-master.config-files.sha256"
export RUN_ROOT="$FACE_DESTYLE_ROOT/outputs/multistyle-routing-validation-137-v1"

cd "$REPO"
git pull --ff-only origin main
```

If GitHub access times out, use AutoDL's official accelerator before the pull:

```bash
source /etc/network_turbo
git pull --ff-only origin main
unset http_proxy
unset https_proxy
```

Run Stage 1 in a `screen` session:

```bash
screen -S multistyle-validation-stage1
```

```bash
cd "$REPO"
mkdir -p "$RUN_ROOT/stage1/images"
set -o pipefail
python scripts/run_flux_kontext_probe.py \
  --manifest data/manifests/multistyle-routing/non_origami_validation_137.jsonl \
  --data-root "$DATA_ROOT" \
  --split extension \
  --required-style comic \
  --required-style ink \
  --required-style watercolor \
  --required-style 3d_cartoon \
  --required-style clay \
  --required-style needle_felt \
  --probe-stage batch \
  --prompt-stage stage1 \
  --model-dir "$MODEL_DIR" \
  --download-manifest "$MODEL_MANIFEST" \
  --hash-manifest "$MODEL_HASHES" \
  --output-dir "$RUN_ROOT/stage1/images" \
  --records-output "$RUN_ROOT/stage1/records.jsonl" \
  --failures-output "$RUN_ROOT/stage1/failures.jsonl" \
  --styles-config configs/styles.yaml \
  --seed 42 \
  --num-inference-steps 28 \
  --guidance-scale 2.5 \
  2>&1 | tee "$RUN_ROOT/stage1/generate.log"
echo "VALIDATION_STAGE1_EXIT_CODE=${PIPESTATUS[0]}"
```

Expected Stage 1 result: 137 records, 137 PNGs, zero failures. If interrupted, rerun the same
command with `--resume`.

Only after Stage 1 completes, run true sequential Stage 2 in another `screen` session:

```bash
screen -S multistyle-validation-stage2
```

```bash
cd "$REPO"
mkdir -p "$RUN_ROOT/stage2-sequential/images"
set -o pipefail
python scripts/run_flux_kontext_probe.py \
  --input-records "$RUN_ROOT/stage1/records.jsonl" \
  --required-style comic \
  --required-style ink \
  --required-style watercolor \
  --required-style 3d_cartoon \
  --required-style clay \
  --required-style needle_felt \
  --probe-stage batch \
  --prompt-stage stage2 \
  --model-dir "$MODEL_DIR" \
  --download-manifest "$MODEL_MANIFEST" \
  --hash-manifest "$MODEL_HASHES" \
  --output-dir "$RUN_ROOT/stage2-sequential/images" \
  --records-output "$RUN_ROOT/stage2-sequential/records.jsonl" \
  --failures-output "$RUN_ROOT/stage2-sequential/failures.jsonl" \
  --styles-config configs/styles.yaml \
  --seed 42 \
  --num-inference-steps 28 \
  --guidance-scale 2.5 \
  2>&1 | tee "$RUN_ROOT/stage2-sequential/generate.log"
echo "VALIDATION_STAGE2_EXIT_CODE=${PIPESTATUS[0]}"
```

Do not add `--manifest` or `--data-root` to Stage 2. Expected result: another 137 records and PNGs,
with the Stage 1 outputs recorded as explicit inputs.

## Completed review and stopping result

The returned run was reviewed through 23 source/Stage1/Stage2 comparison pages. All 137 sources now
have an explicit terminal decision: 71 selected successes and 66 explicit failures. Stage 2 added
five strict Clay rescues and no strict rescue in the other five styles. The final report and route
are in [`results/multistyle_project_closure_20260826.md`](results/multistyle_project_closure_20260826.md)
and [`../configs/multistyle_routing.yaml`](../configs/multistyle_routing.yaml).

No experiment is left active. A new LoRA, teacher run, source-generation cycle, or additional
large-batch validation requires a new operator decision.
