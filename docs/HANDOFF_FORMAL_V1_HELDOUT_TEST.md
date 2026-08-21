# Formal-v1 held-out test handoff and frozen analysis plan

## Start here

This document is the execution contract for the next task. Read `AGENTS.md`,
`docs/research_context.md`, `docs/evaluation_protocol.md`,
`docs/results/formal_v1_calibration_metrics_20260820.md`, and
`docs/results/formal_v1_calibration_human_round_a_20260821.md` before acting.

The commit containing this document freezes the held-out plan before any FLUX test generation or
test-image viewing. The four SDXL runs already contain test outputs because calibration and test
were generated in one model-loading session, but their test records and images remain unreviewed and
unevaluated. Do not open unblinded run contact sheets, raw test outputs, private method keys, or
metric summaries until the relevant blinded scores are frozen. Method-hidden source/candidate pairs
are the only test images the reviewer may open during scoring.

## State inherited from calibration

- Formal-v1 has 120 frozen sources: 20 pilot, 40 calibration, and 60 test; each split is balanced
  across `3d_cartoon`, `comic`, `ink`, and `watercolor`.
- Four SDXL methods have 100 calibration+test outputs each: generic prompt-only, adaptive
  prompt-only, Global Canny 0.4, and Region Canny. All are seed 42 and have zero recorded generation
  failures.
- FLUX Kontext has 40 calibration outputs at native 1024, seed 42, 28 steps, guidance 2.5, and zero
  recorded generation failures. It does not yet have formal-v1 test outputs.
- Calibration automatic metrics completed 200/200 DINO and CLIP pairs. ArcFace produced 197 cosine
  values and three explicit generated-output no-face statuses.
- One 200-pair calibration blind round was frozen before unblinding. FLUX passed 37/40 (92.5%);
  Global Canny and Region Canny each passed 3/40, generic 2/40, and adaptive 1/40.
- None of the four SDXL candidates rescued the three FLUX calibration failures. The formal-v1
  progressive fallback result is therefore frozen as negative: existing SDXL fallback adds zero
  accepted calibration sources after FLUX.
- Calibration Round B was waived before unblinding. Blank manual failure subtypes mean
  `not_reported`, not no failure.
- Qwen2.5-VL-3B is not rerun broadly because the pilot showed method-insensitive score compression.
- LoRA, Multi-ControlNet, Pose, Depth, new prompts, new seeds, and new generators are outside
  formal-v1. Any later 3D-cartoon repair is a separately split `v2` study.

Local returned archives are stored below
`/Users/pot/Documents/大创/实验归档/returned-runs/`; their checksums are listed in
`SHA256SUMS`. Raw images, private method keys, embeddings, model weights, and bulk outputs must not
enter Git.

## Frozen hypotheses and outcomes

### Primary hypothesis

FLUX Kontext has a higher held-out human pass rate than each of the four frozen SDXL baselines on
the same 60 test sources.

### Primary outcome

One method-hidden human pass decision per method-source candidate. A candidate passes only when all
of the following hold:

- content preservation >= 4;
- style removal >= 4;
- facial identity is judgeable; and
- recoverable facial identity >= 4.

An unjudgeable face is a failure and is also counted separately. Missing core human scores invalidate
that candidate; they are never imputed. Blank failure-subtype fields mean `not_reported` and do not
mean no failure.

### Secondary outcomes

- the three human score distributions and means;
- pass counts/rates by the four frozen styles;
- DINOv2 Base, CLIP ViT-L/14, and paired ArcFace scores and explicit no-face statuses;
- deterministic dimension failures derived from the core rubric:
  `content_score < 4`, `style_removal_score < 4`, and invalid or `< 4` identity;
- manually selected failure subtypes only on rows where they were reported;
- agreement from the frozen test-retest subset;
- diagnostic association between automatic metrics and human judgments.

## Frozen methods and settings

Do not change models, revisions, prompts, preprocessing, scheduler, seed, resolution, steps,
guidance, Canny settings, parsing settings, rubric wording, pass thresholds, missing-value handling,
or method labels.

The only remaining generation is the FLUX 60-source test batch:

- backend: `flux1_kontext_dev_prompt_edit_bf16_offloaded`;
- official model: `black-forest-labs/FLUX.1-Kontext-dev` from the existing verified local
  Diffusers-format directory;
- original BF16 weights with model CPU offload;
- frozen adaptive style instructions from `configs/styles.yaml`;
- seed 42;
- native 1024x1024;
- 28 inference steps;
- guidance 2.5;
- batch size 1;
- no Canny, Pose, Depth, LoRA, quantization, second edit, or style-specific tuning.

The FLUX expansion decision is calibration-informed because no numerical expansion threshold was
preregistered earlier. Report it that way; do not retroactively call it a passed preregistered gate.

## Work order and sealing rules

1. Pull the handoff commit and confirm a clean tree. Do not inspect any test output.
2. Before GPU work, implement or verify with synthetic tests:
   - exact test-record selection and five-method pairing;
   - machine-only FLUX test archive validation;
   - blinded review construction for 300 primary candidates plus the frozen repeat subset;
   - held-out statistics described below.
3. Run Ruff, full pytest, every changed script's `--help`, and `git diff --check`; commit and push.
4. On the sole remaining `3l8` host, verify GPU/model/data assets and generate FLUX test once.
5. Do not display output images. Machine-validate records and images, then package without cleanup.
6. Return ZIP plus SHA-256 sidecar; locally verify checksum and ZIP CRC before any server deletion.
7. Build the blind review locally from the five returned/frozen runs. Keep the private key closed.
8. Complete the full blind round, then the delayed repeat subset. Freeze both score files before
   unblinding.
9. Unblind, run the frozen statistics once, create figures/tables, and write the held-out report.
10. Only after that may accepted triplets be constructed or a separately named v2 extension begin.

## FLUX test generation template for `3l8`

Confirm the exact acquisition-manifest names on the host before executing. Expected paths from the
calibration run are shown below; do not invent replacements when a file is missing.

```bash
set -euo pipefail
conda activate face-destyle

export FACE_DESTYLE_ROOT=/root/autodl-tmp/face-destyle
export FACE_DESTYLE_DATA_ROOT="$FACE_DESTYLE_ROOT/data/Face-DeStyle-Data"
export HF_HOME="$FACE_DESTYLE_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="$FACE_DESTYLE_ROOT/cache/torch"
unset OMP_NUM_THREADS

export REPO="$FACE_DESTYLE_ROOT/code/Face-DeStyle-Pipeline"
export MODEL_DIR="$FACE_DESTYLE_ROOT/models/diffusion/FLUX.1-Kontext-dev"
export MODEL_MANIFEST="$FACE_DESTYLE_ROOT/models/download-manifests/flux1-kontext-dev-modelscope-master.txt"
export MODEL_HASHES="$FACE_DESTYLE_ROOT/models/download-manifests/flux1-kontext-dev-modelscope-master.config-files.sha256"
export RUN="$FACE_DESTYLE_ROOT/outputs/formal-v1-test-flux-native1024-seed42"
export ARCHIVE="$FACE_DESTYLE_ROOT/packages/formal-v1-test-flux-native1024-seed42.zip"

cd "$REPO"
source /etc/network_turbo
git pull --ff-only origin main
git log -1 --oneline

nvidia-smi
python -c 'import torch; print("CUDA available:", torch.cuda.is_available())'
test -d "$MODEL_DIR"
test -f "$MODEL_MANIFEST"
test -f "$MODEL_HASHES"
test -d "$FACE_DESTYLE_DATA_ROOT"

test ! -e "$RUN" || {
  echo "Run directory already exists; inspect records and use --resume only for a documented interruption: $RUN"
  exit 1
}
mkdir -p "$RUN/images"

python scripts/run_flux_kontext_probe.py \
  --manifest data/manifests/formal-v1/inputs.jsonl \
  --data-root "$FACE_DESTYLE_DATA_ROOT" \
  --split test \
  --probe-stage batch \
  --model-dir "$MODEL_DIR" \
  --download-manifest "$MODEL_MANIFEST" \
  --hash-manifest "$MODEL_HASHES" \
  --output-dir "$RUN/images" \
  --records-output "$RUN/records.jsonl" \
  --failures-output "$RUN/failures.jsonl" \
  --seed 42 \
  --num-inference-steps 28 \
  --guidance-scale 2.5 \
  2>&1 | tee "$RUN/generate.log"

python scripts/package_run.py \
  --run-dir "$RUN" \
  --archive "$ARCHIVE"

cd "$FACE_DESTYLE_ROOT/packages"
sha256sum -c "$(basename "$ARCHIVE").sha256"
```

If interrupted, first verify that existing `records.jsonl` entries have unique test source IDs and
present outputs; then repeat the same runner command with `--resume`. Never delete or overwrite a
partial run merely to obtain a clean-looking log. Do not pass `--cleanup` before local verification.

## Required machine validation of the returned FLUX test

Without displaying images, verify and record:

- ZIP sidecar matches and `ZipFile.testzip()`/`unzip -t` passes;
- exactly 60 unique records and exactly the frozen 60 test source IDs;
- no calibration or pilot source ID appears;
- every record uses seed 42 and the frozen backend/settings;
- output basenames are unique and every declared output exists in the archive;
- 60 main outputs decode as RGB 1024x1024;
- failures are absent or explicitly recorded and counted;
- no output was overwritten and no duplicate record exists.

File integrity is not human quality evidence. Do not create or view a contact sheet during this
validation.

## Frozen test blind-review design

Default to a single-rater blinded evaluation because that matches available project resources:

- primary round: all 300 candidates, 60 sources x five methods;
- repeat round: exactly 60 candidates (20%), selected before scoring by stratifying on method and
  style—three of the 15 candidates in each of 5 x 4 method-style cells;
- primary randomization seed: `20260821`;
- repeat selection and ordering seed: `20260822`;
- hide method, source ID, automatic metrics, generation metadata, and filenames carrying method
  identity;
- show source and candidate at equal displayed size;
- complete and freeze the primary round before opening the repeat round;
- keep the private key closed until both score files are frozen;
- no adjudication after unblinding for a single-rater protocol.

Report test-retest quadratic-weighted Cohen kappa for each 0--5 dimension, unweighted Cohen kappa
for pass/fail, exact agreement, mean absolute score difference, and the number of repeated pairs.
Call this single-rater test-retest agreement, not inter-rater agreement. If a second independent
rater becomes available, changing to a two-rater design requires a committed protocol amendment
before either person views a test pair; do not switch after scoring begins.

## Frozen statistical analysis

All five methods are paired on the same 60 `source_id` values.

### Primary comparisons

- Compare FLUX separately with each of the four SDXL baselines using the exact paired McNemar test
  on pass/fail.
- Use two-sided tests and Holm correction across the four baseline comparisons.
- Report the paired pass-rate difference and a 95% source-level paired bootstrap confidence
  interval using 20,000 resamples and seed `20260821`.
- Report each method's pass count/rate and Wilson 95% binomial interval.
- The overall primary hypothesis is supported only if FLUX exceeds all four baselines with positive
  paired-difference intervals and Holm-adjusted `p < 0.05`; otherwise report the exact mixed result.

### Ordinal human scores

- Compare FLUX with each baseline using two-sided paired Wilcoxon signed-rank tests for content,
  style removal, and identity.
- Apply Holm correction across the four comparisons separately within each score dimension.
- Report raw distributions, means, medians, paired differences, and 95% source bootstrap intervals;
  do not report only p-values.
- Scores with invalid identity are excluded from ordinal identity comparisons but remain failed
  candidates in pass-rate analysis.

### Style analysis

Report pass counts/rates and score distributions for the four frozen styles, 15 sources per cell.
Treat these as planned descriptive subgroup analyses; do not multiply unplanned subgroup tests.
Check whether the calibration pattern of 3D-cartoon material/geometry difficulty recurs without
assuming that it must.

### Metric-alignment diagnostics

- Spearman: DINO and CLIP versus human content; ArcFace versus human identity; each automatic metric
  versus human style removal.
- ROC-AUC of each automatic metric for human pass/fail, as a diagnostic only.
- ArcFace no-face values remain missing with their status counted; do not impute zero. Use complete
  cases for correlation/AUC and state `n`.
- Report pooled correlations plus per-method descriptive sensitivity results so that method
  differences are not mistaken for within-method validity.
- Do not construct a composite metric or select acceptance thresholds from held-out test.

## Frozen routing and reporting decisions

- FLUX is the primary formal-v1 method.
- A failed FLUX candidate is rejected. The existing SDXL candidates are comparison baselines, not
  fallbacks, because they rescued 0/3 calibration FLUX failures.
- Report all attempted sources and failures; do not cherry-pick examples.
- Primary test reporting uses the complete 60-source set. No method is removed after results.
- The calibration and test conclusions remain separate. Pilot is debugging evidence only.
- Any post-test 3D-cartoon repair uses a new development set and independent test set under a v2
  label; it cannot revise formal-v1.
- After test analysis, accepted triplets must preserve split labels and select same-style,
  different-source references from within the same split.

Required final artifacts are: method pass-rate table with 95% intervals, per-style table/plot,
paired-test table with Holm adjustment, score distributions, metric-alignment table/heatmap,
test-retest agreement, complete failure counts, and non-cherry-picked representative success/failure
panels selected by frozen rules stated before viewing candidate identities.

## Stop and ask conditions

Stop rather than improvise if the formal manifest differs from 60 test sources, a run path already
contains unexplained files, the model/acquisition manifests differ from calibration, any setting
would change, a returned archive fails integrity checks, blind IDs or method-source pairs are
missing/duplicated, or scoring begins before the key can be sealed. Do not solve these conditions by
regenerating selected samples, changing seeds, relaxing thresholds, or opening test images.

## Known operational pitfalls from earlier runs

These are established project facts. Diagnose them explicitly instead of ending the task after the
first failed command.

- **Local Mac is not AutoDL.** Codex normally runs in `/Users/pot/Github/Face-DeStyle-Pipeline` on
  macOS. Never execute `/root/autodl-tmp/...`, `nvidia-smi`, CUDA inference, or the server generation
  block locally. Locally inspect/edit/test/push; give the server block to the operator to run on
  `3l8`.
- **Do not use one opaque fail-fast block for discovery.** Run read-only preflight checks first and
  print which path/check failed. `set -euo pipefail` is suitable only after paths are resolved. A
  missing optional file or an inactive GPU is a diagnosis to report, not permission to terminate
  without examining the state.
- **Python command differs by environment.** Local validation uses
  `conda run -n face-destyle python ...`; bare `python` may not exist on macOS. AutoDL uses `python`
  only after `conda activate face-destyle`.
- **GitHub needs the AutoDL proxy.** Use `source /etc/network_turbo` for `git pull --ff-only origin
  main`. Do not print proxy values or credentials. Proxy variables may be unset after the pull when
  they interfere with local-only model operations.
- **`OMP_NUM_THREADS` was malformed before.** Always `unset OMP_NUM_THREADS`; the earlier
  `libgomp: Invalid value` warning was environment configuration, not a model failure.
- **The manifest loader validates the whole 120-source manifest.** Even with `--split test`, the
  supplied `--data-root` must resolve every frozen pilot/calibration/test asset and SHA-256. The
  correct common root is normally `$FACE_DESTYLE_ROOT/data/Face-DeStyle-Data`, containing legacy
  `raw/` plus `batch1/` and `batch2/`. Do not conclude that test is missing from one unresolved path.
- **Test IDs and filenames differ.** Test manifest record IDs/output basenames normally carry a
  `test-` prefix while `source_id` may not. Validate an output by the basename declared in
  `record.output_path`, never by assuming `SOURCE_ID.png`; this exact assumption previously caused
  a false missing-image report.
- **The FLUX acquisition manifest is intentionally lightweight.** The operator chose not to run a
  full 54 GB SHA-256. The download manifest records
  `large_weight_sha256=not_run_operator_choice`; the config-file hash manifest plus successful BF16
  load/inference are the accepted integrity evidence. Do not block test on a new full-weight hash.
- **Resolve the exact manifest filenames instead of inventing them.** Expected files are
  `flux1-kontext-dev-modelscope-master.txt` and
  `flux1-kontext-dev-modelscope-master.config-files.sha256`, but use read-only `find`/`ls` if a
  preflight path fails. Never substitute another FLUX repository or quantized copy.
- **FLUX is native 1024.** Diffusers previously adjusted non-native dimensions to 1024. Formal-v1
  intentionally compares FLUX 1024 with SDXL 768; do not resize FLUX to imitate SDXL or call equal
  seeds matched noise across architectures.
- **Generation is slow but normal.** Frozen FLUX inference has taken roughly 90--107 seconds per
  image with model offload. Sixty images can require around 1.5--2 hours plus loading/packaging. A
  progress bar moving slowly is not a hang; maintain checkpoint records and communicate status.
- **An existing run directory is not automatically corruption.** Inspect `records.jsonl`,
  `failures.jsonl`, output paths, source IDs, and frozen settings. Use the hardened `--resume` only
  for a valid interrupted run; if 60 successes already exist, validate/package rather than rerun.
  Never overwrite or delete the directory just to restart cleanly.
- **Zero failures may mean no `failures.jsonl`.** Absence of that optional file is valid when no
  failure was recorded. If present, it must be parsed and counted.
- **Package and verify before cleanup.** `package_run.py` refuses overwrite and writes a sidecar
  using the archive basename. Run `sha256sum -c` from the package directory, transfer ZIP and
  sidecar, verify locally, and omit `--cleanup` until that succeeds.
- **Machine validation is not visual inspection.** Pillow decode, dimensions, mode, record parsing,
  CRC, and SHA checks are allowed while test is sealed. `view_image`, screenshots, raw output
  browsing, unblinded contact sheets, and metric-result inspection are not.
- **ArcFace provider warnings are not a rerun trigger.** ONNX Runtime previously exposed only CPU
  and fell back from requested CUDA. That changes runtime, not the paired metric definition. Record
  explicit no-face status and do not silently convert it to zero.
- **Do not redo completed tooling.** Commit `441640a` added the held-out archive validator, blind
  builder, statistical analyzer, and hardened FLUX resume checks. Inspect and test these files;
  preserve them unless a concrete bug is demonstrated.
