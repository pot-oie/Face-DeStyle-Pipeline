# Material-extension-v1 progressive destylization experiment

## Purpose

This is a small exploratory extension designed to turn the face-domain findings into a complete
progressive research story. It is independent of formal-v1 and must not revise or be pooled with
the formal-v1 held-out result.

The experiment asks:

1. Can a fixed material-aware Stage 2 instruction rescue 3D-cartoon and Clay failures left by the
   general FLUX instruction?
2. Does Needle-felt retain fibrous material after both prompt stages while preserving enough facial
   evidence to justify a specialized destylization LoRA?
3. Can a structured material-aware scorer route candidates to accept, retry, LoRA, or reject without
   using one opaque aggregate score?

## Validated source matrix

Use exactly 15 exploratory sources:

- five existing `3d_cartoon` sources from the formal-v1 `pilot` split:
  `synthetic-3d-cartoon-001` through `synthetic-3d-cartoon-005`;
- five Clay sources from
  `extensions/material_styles_v1/manifests/extension_pilot_inputs.jsonl`;
- five Needle-felt sources from the same extension manifest.

The material extension contains 24 candidates: 12 Clay and 12 Needle-felt. Input-only QC accepted
15 and selected five per style for the pilot. Independent machine validation confirmed that the ten
selected files are unique, decodable RGB images with shortest side at least 1254 pixels, have
complete accepted provenance, and have no source ID, source group, or file overlap with formal-v1.

The selected Clay inputs are real CC0 terracotta/earthenware museum objects, whereas Needle-felt
inputs are fictional project-generated portraits. Clay therefore tests a harder reconstruction
boundary: missing hair, body, or natural skin must not be treated as recoverable real identity.
Identity means consistency with the depicted facial evidence only.

## Frozen generation comparison

### Stage 1: general FLUX destylization

- model: local original-BF16 `FLUX.1-Kontext-dev`;
- model CPU offload, batch size 1;
- native 1024x1024;
- seed 42;
- 28 steps;
- guidance 2.5;
- `stage1_prompt` from `configs/styles.yaml`;
- original source image as the edit input;
- no Canny, Pose, Depth, LoRA, quantization, or second edit.

Reuse the already completed five-source 3D pilot Stage 1 run. Generate only the ten new Clay and
Needle-felt Stage 1 candidates.

### Stage 2: fixed material-aware regeneration

Run the same 15 original source images again with the corresponding frozen `stage2_prompt`. Stage 2
is not an edit of the Stage 1 output; it is a new edit from the original material-style source. Hold
model, seed, resolution, steps, guidance, and all other settings constant. The only changed factor is
the declared instruction.

This yields a paired 15-source comparison: Stage 1 versus Stage 2, five sources per style and 30
candidate outputs total.

## Material-aware scorer

Use `configs/material_extension_scorer_prompt.txt` and
`configs/material_extension_evaluation.yaml`. The scorer returns four 0-5 dimensions:

- content preservation;
- style removal;
- recoverable facial identity preservation plus judgment validity;
- material removal.

It also reports explicit residual types such as CGI rendering, plastic skin, clay/ceramic surface,
wool fiber, fuzzy textile, handmade-doll geometry, and synthetic lighting. Do not construct a
composite score. A candidate passes only when all four dimensions are at least 4 and identity is
judgeable.

The structured VLM output is a routing proposal, not ground truth. Complete one human review of all
30 Stage 1/Stage 2 candidates with stage identity hidden, then compare VLM routes with the human
decision. Report disagreements rather than silently overriding them.

Routing is deterministic:

- Stage 1 passes all gates: `accept`;
- Stage 1 preserves content and identity but fails style/material removal: `retry_prompt`;
- Stage 2 preserves content and identity but still fails style/material removal: `route_lora`;
- unjudgeable identity or failed content/identity: `reject`.

## Stage 3 LoRA feasibility gate

Do not train a LoRA merely because the extension includes material styles. A style becomes eligible
only if at least three of its five Stage 2 candidates are `route_lora`: content and recoverable
identity both pass, while style or material removal still fails. This separates a learnable material
residual from an ill-posed input whose identity evidence is already absent.

If eligible, first run a 20-30-pair overfit smoke test for a reverse edit LoRA:

- input: synthetic Clay or Needle-felt portrait;
- instruction: restore a natural photographic portrait;
- target: the unchanged natural portrait used to synthesize that material-style input.

Only after the overfit test visibly reduces the intended material without identity collapse should
the training set expand to roughly 100-200 pairs. Training pairs must be separate from the 15-source
evaluation matrix. Report this as a small feasibility prototype, not large-scale training.

## Outcomes

For each style and stage, report:

- pass count out of five;
- Stage 2 rescue count among Stage 1 failures;
- content, style-removal, identity, and material-removal score distributions;
- residual-type counts;
- VLM/human routing agreement;
- rejected inputs with the reason `unrecoverable_identity_or_content`;
- if LoRA runs, additional rescue count and any new identity failures.

The central result is the progressive path, including negative outcomes:

`general FLUX -> material-aware regeneration -> specialized LoRA when justified -> reject when the
source does not support reliable reconstruction`.

## Machine validation command

```bash
python scripts/validate_material_extension.py \
  --data-root /root/autodl-tmp/face-destyle/data/Face-DeStyle-Data \
  --manifest /root/autodl-tmp/face-destyle/data/Face-DeStyle-Data/extensions/material_styles_v1/manifests/extension_pilot_inputs.jsonl \
  --provenance /root/autodl-tmp/face-destyle/data/Face-DeStyle-Data/extensions/material_styles_v1/manifests/extension_pilot_provenance.jsonl \
  --formal-manifest data/manifests/formal-v1/inputs.jsonl
```

Generation commands should use `scripts/run_flux_kontext_probe.py` with `--probe-stage batch`, the
appropriate `--prompt-stage`, and explicit repeated `--required-style` arguments. Run Stage 1 and
Stage 2 in separate output directories and keep their records separate. Use `tmux`, capture the
runner's real exit code through `${PIPESTATUS[0]}`, and never start a second runner before checking
the first process.

## AutoDL run template

Before running, copy the complete `extensions/material_styles_v1` directory to the AutoDL data root
without changing relative paths. Activate the existing frozen environment, `unset OMP_NUM_THREADS`,
and confirm no existing runner or unexplained output directory. Do not install or upgrade packages.

Set these paths once inside a persistent `tmux` session:

```bash
set +e
export FACE_DESTYLE_ROOT=/root/autodl-tmp/face-destyle
export REPO="$FACE_DESTYLE_ROOT/code/Face-DeStyle-Pipeline"
export DATA_ROOT="$FACE_DESTYLE_ROOT/data/Face-DeStyle-Data"
export MODEL_DIR="$FACE_DESTYLE_ROOT/models/diffusion/FLUX.1-Kontext-dev"
export MODEL_MANIFEST="$FACE_DESTYLE_ROOT/models/download-manifests/flux1-kontext-dev-modelscope-master.txt"
export MODEL_HASHES="$FACE_DESTYLE_ROOT/models/download-manifests/flux1-kontext-dev-modelscope-master.config-files.sha256"
unset OMP_NUM_THREADS
cd "$REPO" || exit 1
pgrep -af run_flux_kontext_probe.py || true
```

Generate the ten new Stage 1 material candidates:

```bash
export RUN="$FACE_DESTYLE_ROOT/outputs/material-extension-v1-stage1-flux-native1024-seed42"
mkdir -p "$RUN/images"
python scripts/run_flux_kontext_probe.py \
  --manifest "$DATA_ROOT/extensions/material_styles_v1/manifests/extension_pilot_inputs.jsonl" \
  --data-root "$DATA_ROOT" \
  --split extension \
  --required-style clay \
  --required-style needle_felt \
  --probe-stage batch \
  --prompt-stage stage1 \
  --model-dir "$MODEL_DIR" \
  --download-manifest "$MODEL_MANIFEST" \
  --hash-manifest "$MODEL_HASHES" \
  --output-dir "$RUN/images" \
  --records-output "$RUN/records.jsonl" \
  --failures-output "$RUN/failures.jsonl" \
  --styles-config configs/styles.yaml \
  --seed 42 \
  --num-inference-steps 28 \
  --guidance-scale 2.5 2>&1 | tee "$RUN/generate.log"
runner_rc=${PIPESTATUS[0]}
echo "MATERIAL_STAGE1_EXIT_CODE=$runner_rc"
```

After Stage 1 succeeds and its records are machine-checked, use the same command with a separate
directory named `material-extension-v1-stage2-flux-native1024-seed42` and change only
`--prompt-stage stage1` to `--prompt-stage stage2`.

Generate the five 3D Stage 2 candidates from the original formal pilot sources:

```bash
export RUN="$FACE_DESTYLE_ROOT/outputs/material-extension-v1-stage2-3d-flux-native1024-seed42"
mkdir -p "$RUN/images"
python scripts/run_flux_kontext_probe.py \
  --manifest data/manifests/formal-v1/inputs.jsonl \
  --data-root "$DATA_ROOT" \
  --split pilot \
  --required-style 3d_cartoon \
  --probe-stage batch \
  --prompt-stage stage2 \
  --model-dir "$MODEL_DIR" \
  --download-manifest "$MODEL_MANIFEST" \
  --hash-manifest "$MODEL_HASHES" \
  --output-dir "$RUN/images" \
  --records-output "$RUN/records.jsonl" \
  --failures-output "$RUN/failures.jsonl" \
  --styles-config configs/styles.yaml \
  --seed 42 \
  --num-inference-steps 28 \
  --guidance-scale 2.5 2>&1 | tee "$RUN/generate.log"
runner_rc=${PIPESTATUS[0]}
echo "MATERIAL_3D_STAGE2_EXIT_CODE=$runner_rc"
```

If a runner returns nonzero, stop and inspect its log, records, failures, GPU process, and system
memory. Do not immediately repeat it. Use `--resume` only after the hardened state validator accepts
the existing run directory.
