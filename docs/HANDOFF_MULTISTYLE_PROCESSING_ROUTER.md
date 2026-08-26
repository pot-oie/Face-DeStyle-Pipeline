# Multistyle processing-router completion handoff

## Active task

Complete the missing Stage 1/true-sequential-Stage 2 evidence and then implement a manifest-driven
processing router. This supersedes the statement that the entire multistyle branch was closed. The
LoRA branch remains closed: do not restart Origami V2/V2.1, 3D style-contrast19, or any multistyle
adapter.

The factual coverage audit is
`docs/results/multistyle_processing_coverage_audit_20260826.md`. The frozen 24-source manifest is
`data/manifests/multistyle-routing/missing_stage12_sources.jsonl`.

## Operator execution preference

Keep this experiment practical. Use AutoDL's official clone/migration workflow and official
academic network accelerator when needed. Do not add archive hashing, weight hashing, blind review,
formal acceptance gates, repeat scoring, or broad integrity audits. The manifest loader's normal
file-presence/checksum validation is enough for the 24 small input files; do not hash returned
images, model shards, caches, or the complete data disk.

The only pre-run checks required are:

- the cloned system and data disks contain the existing repository, environment, model, and input
  root;
- `nvidia-smi` and `torch.cuda.is_available()` show the selected GPU;
- no duplicate Kontext runner is active;
- the new output directories do not already contain unexplained files.

For GitHub or Hugging Face access, AutoDL's official terminal command is:

```bash
source /etc/network_turbo
```

Disable it when it interferes with normal access:

```bash
unset http_proxy
unset https_proxy
```

Do not print proxy values, tokens, SSH endpoints, or other credentials into tracked logs.

## Instance clone and GPU choice

The current work is BF16 Kontext inference only, with CPU/model offload and batch size 1; it does
not train a LoRA. The prior real run reached about 24.9 GB peak allocated VRAM on an RTX 4080 SUPER
vGPU with 32 GB visible memory. Therefore:

1. prefer `vGPU-32GB` for the cloned instance; it matches the proven inference envelope and avoids
   paying for unused 48 GB capacity;
2. use RTX 5090 32GB only if the cloned PyTorch/CUDA environment loads it without an upgrade;
3. avoid 24GB RTX 4090/4090D/3090 for the unchanged BF16 path because the recorded peak leaves less
   than 1 GiB of nominal headroom before framework and transient allocations;
4. do not choose 12--16GB cards, dual-3080, H800, or 96GB professional cards for this run.

Follow AutoDL's official same-region clone flow: power off the source instance, choose clone, and
include the data disk. If the data disk was not selected during clone, use AutoDL's cross-instance
data-disk copy rather than reconstructing the environment or redownloading weights. Do not delete
or overwrite the old instance until the cloned environment has completed one real Stage 1 image.

## Phase 1: missing four-style generation

No AutoDL run has been started by this handoff. Once the operator creates and opens the cloned
instance, use the existing environment and original BF16 Kontext model. Do not reinstall or
download weights.

Expected paths:

```bash
export FACE_DESTYLE_ROOT=/root/autodl-tmp/face-destyle
export REPO="$FACE_DESTYLE_ROOT/code/Face-DeStyle-Pipeline"
export DATA_ROOT="$FACE_DESTYLE_ROOT/data/Face-DeStyle-Data"
export MODEL_DIR="$FACE_DESTYLE_ROOT/models/diffusion/FLUX.1-Kontext-dev"
export MODEL_MANIFEST="$FACE_DESTYLE_ROOT/models/download-manifests/flux1-kontext-dev-modelscope-master.txt"
export MODEL_HASHES="$FACE_DESTYLE_ROOT/models/download-manifests/flux1-kontext-dev-modelscope-master.config-files.sha256"
export RUN_ROOT="$FACE_DESTYLE_ROOT/outputs/multistyle-routing-gap-v1"
```

Stage 1 runs all 24 sources:

```bash
cd "$REPO"
mkdir -p "$RUN_ROOT/stage1/images"
python scripts/run_flux_kontext_probe.py \
  --manifest data/manifests/multistyle-routing/missing_stage12_sources.jsonl \
  --data-root "$DATA_ROOT" \
  --split extension \
  --required-style comic \
  --required-style ink \
  --required-style watercolor \
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
  --guidance-scale 2.5
```

Only after Stage 1 returns 24 successful records, run the true chain:

```bash
mkdir -p "$RUN_ROOT/stage2-sequential/images"
python scripts/run_flux_kontext_probe.py \
  --input-records "$RUN_ROOT/stage1/records.jsonl" \
  --required-style comic \
  --required-style ink \
  --required-style watercolor \
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
  --guidance-scale 2.5
```

Do not use `--manifest` for Stage 2. That would re-edit the original source and repeat the old
Needle-felt design error.

## Phase 2: Origami fallback probe

After the four-style review, separately test `matv2-origami-002`, `011`, and `018` as:

`original source -> frozen V1 checkpoint 100 -> true residual-material Stage 2`

This is not a new adapter run. Preserve the fixed V1 weight and settings. Prepare exact prompts and
commands only after confirming the V1 output records and weight remain available on AutoDL.

## Phase 3: executable router

Implement a lightweight, human-review-driven router rather than an untrained automatic selector:

1. plan Stage 1 from input style;
2. emit a review CSV;
3. accept explicit decisions such as `accept_stage1`, `run_stage2`, `run_origami_v1`,
   `route_teacher`, or `explicit_failure`;
4. run only the selected next stage without overwriting prior outputs;
5. preserve parent-record provenance and write the terminal route for every source.

Teacher generation remains an external, manually authorized fallback. The router must never report
an unavailable teacher or adapter result as successful.
