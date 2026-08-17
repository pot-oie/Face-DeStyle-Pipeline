# Formal evaluation and next-experiment handoff

## What is already downloaded

The 2026-08-14 offline inventory found every primary evaluation asset complete on the AutoDL data
disk: DINOv2 Base, CLIP ViT-L/14, InsightFace buffalo_l, and Qwen2.5-VL-3B. It also found the
robustness assets DINOv2 Large, SigLIP SO400M, Qwen2.5-VL-7B, and Florence-2 Large. FLUX.1
Kontext-dev was downloaded later and completed a 20/20 BF16 run. “Downloaded” means the required
files existed and their containers passed the recorded checks; only FLUX, SDXL, Canny, and face
parsing have completed real inference in this project so far.

Reconfirm the four primary evaluators without checking unrelated extensions:

```bash
export FACE_DESTYLE_ROOT=/root/autodl-tmp/face-destyle
export HF_HOME="$FACE_DESTYLE_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
python scripts/check_model_assets.py --config configs/models.yaml \
  --asset dinov2_base \
  --asset clip_vit_l14 \
  --asset insightface_buffalo_l \
  --asset qwen25_vl_3b
```

This command is offline and does not load a model. All four lines must say `OK` before occupying a
GPU. A missing cached model usually means that `HF_HOME` points to the wrong disk, not that the
weights need to be downloaded again.

## One-session primary evaluation

The evaluator consumes any number of `METHOD=records.jsonl` inputs, loads each model only once,
evaluates every supplied pair, releases it before the next large model, and atomically checkpoints
after every pair. It never downloads weights and never writes embeddings. DINO, CLIP, and ArcFace
values are raw cosine similarities; Qwen values are rubric scores from 0 to 5. None are calibrated
acceptance probabilities.

First locate the five frozen run records and set the variables to the exact files printed:

```bash
find "$FACE_DESTYLE_ROOT/outputs" -type f -name records.jsonl -print | sort

export GENERIC_RECORDS=/exact/path/to/generic-strength070/records.jsonl
export ADAPTIVE_RECORDS=/exact/path/to/adaptive-strength070/records.jsonl
export GLOBAL_CANNY_RECORDS=/exact/path/to/global-canny-scale040/records.jsonl
export REGION_CANNY_RECORDS=/exact/path/to/region-canny/records.jsonl
export FLUX_RECORDS=/exact/path/to/flux-kontext-native1024-pilot-20260815-023158/records.jsonl

for path in \
  "$GENERIC_RECORDS" "$ADAPTIVE_RECORDS" "$GLOBAL_CANNY_RECORDS" \
  "$REGION_CANNY_RECORDS" "$FLUX_RECORDS"
do
  test -f "$path" || { echo "missing records: $path"; exit 1; }
done
```

Then run all 100 method/source pairs in one process:

```bash
unset OMP_NUM_THREADS
set -o pipefail
export EVAL_RUN="$FACE_DESTYLE_ROOT/outputs/evaluation-primary-pilot-20260816"
mkdir -p "$EVAL_RUN"

python scripts/evaluate_formal.py \
  --records "prompt_generic=$GENERIC_RECORDS" \
  --records "prompt_adaptive=$ADAPTIVE_RECORDS" \
  --records "global_canny_0p4=$GLOBAL_CANNY_RECORDS" \
  --records "region_canny=$REGION_CANNY_RECORDS" \
  --records "flux_kontext_native1024=$FLUX_RECORDS" \
  --output "$EVAL_RUN/formal-evaluations.jsonl" \
  2>&1 | tee "$EVAL_RUN/evaluate.log"

python scripts/summarize_formal_evaluations.py \
  --evaluations "$EVAL_RUN/formal-evaluations.jsonl" \
  --output-dir "$EVAL_RUN/summary"
```

If the process is interrupted, run the same evaluator command with `--resume`. Add
`--retry-failures` only after reading the recorded failure message. A no-face ArcFace result is
stored as a status and is not silently converted to zero. For provider problems, retry ArcFace
alone with `--metric arcface --arcface-provider cpu --resume --retry-failures`; this uses more wall
time but does not justify stopping the other metrics.

The first SDXL evaluation attempt on Transformers 5.15 completed DINO for 80/80 pairs, ArcFace for
79/80 pairs with one explicit no-face result, and Qwen for 74/80 pairs. CLIP failed on all 80 pairs
because that Transformers version returns a structured pooling object rather than the older tensor.
Six Qwen responses used integral floats or one-character numeric strings instead of JSON integers.
Commit `HEAD` accepts both documented output shapes without relaxing the 0--5 rubric. After pulling
the fix, repeat the same `--records` arguments and retry only the failed metrics:

```bash
python scripts/evaluate_formal.py \
  --records "prompt_generic=$GENERIC_RECORDS" \
  --records "prompt_adaptive=$ADAPTIVE_RECORDS" \
  --records "global_canny_0p4=$GLOBAL_CANNY_RECORDS" \
  --records "region_canny=$REGION_CANNY_RECORDS" \
  --output "$EVAL_RUN/formal-evaluations.jsonl" \
  --metric clip --metric qwen \
  --resume --retry-failures \
  2>&1 | tee "$EVAL_RUN/retry-clip-qwen.log"
```

Do not include FLUX records in this retry unless they were already present in the same evaluation
JSONL. Resume uses the frozen records in that file; the repeated record arguments document the
intended input set.

Package the exact evaluation directory only after inspecting `summary/summary.json`:

```bash
python scripts/package_run.py \
  --run-dir "$EVAL_RUN" \
  --archive "$FACE_DESTYLE_ROOT/packages/evaluation-primary-pilot-20260816.zip"
```

Do not use `--cleanup` until the ZIP and its SHA-256 have been transferred and verified.

## Experiment sequence after the primary evaluation

The order below minimizes new generation and prevents the existing 20-source pilot from being
mistaken for a held-out result.

1. **Freeze the current 100 outputs.** Randomize method labels and have at least two passes of the
   written 0--5 human rubric. Use this already-observed pilot only to debug metric failure handling
   and draft thresholds; do not call it held-out evidence.
2. **Calibrate, then freeze.** Compare DINO/CLIP/ArcFace/Qwen with blinded human labels by style.
   Choose thresholds and missing-face policy on calibration data only. Do not average the metrics
   into an unexplained composite score.
   `scripts/build_blind_review.py --split calibration` builds two independently shuffled,
   method-hidden rounds from the frozen calibration manifest after every method directory is
   complete.
3. **Audit only disagreements.** Run DINOv2 Large, SigLIP, and Qwen 7B only on cases where primary
   metrics disagree with humans or methods are nearly tied. Florence remains a caption/failure audit,
   not another vote. This avoids paying to run every downloaded model on every pair.
4. **Test routing without new generation.** Use the already-generated SDXL candidates to define a
   failure-aware rule on calibration data: prompt-only first; consider Canny only for detected
   structure drift; reject rather than route when style remains or facial evidence is unusable.
   Freeze the rule, then evaluate its accepted-pair yield against each uniform method on held-out
   sources. Candidate reuse makes this a selection experiment, not another generation sweep.
5. **Build the formal source set before renting another long session.** Target 30 authorized,
   source-group-independent images per style: 5 pilot, 10 calibration, and 15 held-out test. The
   current five per style are pilot/debug and cannot also become the formal test set.
6. **Generate the formal SDXL matrix in one rental session.** For new calibration/test sources,
   run the four frozen SDXL methods with one pipeline load per backend and no tuning. These methods
   answer the primary RQs. Estimate runtime from measured means plus load overhead before renting.
7. **Gate formal FLUX expansion.** FLUX is exploratory and costs about 92 seconds per image on the
   measured 4080 SUPER. Run the 40-source calibration portion first. Proceed to the 60-source test
   portion only if the predeclared human/metric criterion is met; otherwise report the 20-source
   capability probe and stop. Keep native 1024 and report the SDXL resolution mismatch.
8. **Open one extension only for a frozen failure class.** Pose, depth, InstantID, RealVisXL, or a
   larger evaluator is justified only when the calibrated failure table names the problem it is
   meant to solve. Do not launch another broad parameter sweep.

The narrow ArcFace operator interpretation in `AGENTS.md` applies throughout: paired drift
diagnostics only, no identity search, enrollment, authentication, surveillance, or publication of
reusable face templates.

## Formal-v1 calibration-only batch on the remaining host

All five formal generation archives were returned and validated locally on 2026-08-17. The next
GPU action is one 200-pair calibration evaluation: 40 frozen calibration sources across the four
SDXL methods and FLUX. Use DINO, CLIP, and paired ArcFace only for this pass. The pilot showed that
Qwen2.5-VL-3B compressed style-removal scores, so do not spend GPU time running it over all formal
pairs; reserve larger-VLM review for disagreements after human labels exist. The preparation script
selects the exact calibration IDs from the complete SDXL record files and refuses missing,
duplicate, mismatched, or nonexistent paths. It does not emit test records.

Use the complete copy-paste batch in the task handoff after pulling the commit that introduced
`scripts/prepare_formal_evaluation_records.py`. Package the evaluation directory without cleanup,
transfer the ZIP and sidecar, and verify them locally before deleting any server output. Build the
blinded review materials off-GPU from the returned archives; do not spend rented GPU time rendering
contact sheets.
