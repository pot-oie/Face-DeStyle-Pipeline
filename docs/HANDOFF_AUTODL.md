# AutoDL handoff — 2026-08-14

## State at handoff

The local working tree now contains a declarative model registry (`configs/models.yaml`), a primary
and extension experiment matrix (`configs/experiments.yaml`), deterministic run-plan expansion,
and an offline asset checker. These additions do not load GPU packages or download weights.

The user-supplied final AutoDL inventory reported:

- 181.63 GiB free on `/root/autodl-tmp`;
- 51.15 GiB of local models and 11.37 GiB of Hugging Face cache;
- 8 cached Hugging Face snapshots and 10 complete HFD manifests;
- complete Qwen2.5-VL 3B and 7B shard sets;
- all five InsightFace buffalo_l ONNX files;
- zero temporary files, broken links, bad HFD metadata, or Safetensors validation failures.

This inventory is a file-integrity result, not a GPU inference result. Prompt-only SDXL and global
Canny later completed smoke runs documented below. Pose, depth, refiner, InstantID, RealVisXL,
DINO/CLIP/SigLIP, ArcFace, Florence, and Qwen metric execution remain unverified.

The repository root `AGENTS.md` is authoritative for the local-macOS versus AutoDL boundary and
the non-secret network/mirror policy. A new task must read it before issuing server commands.

The experiment extension has been reviewed and prepared for publication on `origin/main`. AutoDL
should fast-forward from that branch before beginning GPU work.

## Local validation used before publication

```bash
cd /Users/pot/Github/Face-DeStyle-Pipeline
git status --short
git diff --check
conda run -n face-destyle ruff check .
conda run -n face-destyle pytest
```

## First AutoDL commands after commit and push

```bash
conda activate face-destyle
export FACE_DESTYLE_ROOT=/root/autodl-tmp/face-destyle
unset OMP_NUM_THREADS
export HF_HOME="$FACE_DESTYLE_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="$FACE_DESTYLE_ROOT/cache/torch"

cd "$FACE_DESTYLE_ROOT/code/Face-DeStyle-Pipeline"
git pull --ff-only origin main
python -m pip install -e ".[gpu,dev]"
python scripts/check_environment.py --gpu
python scripts/check_model_assets.py
python scripts/list_experiments.py --seed 42 --json
```

At the start of the server session, explicitly determine whether the instance has a GPU and whether
a proxy/mirror is active:

```bash
nvidia-smi || true
python -c 'import torch; print("CUDA available:", torch.cuda.is_available())'
env | grep -E '^(HF_HOME|HF_HUB_CACHE|HUGGINGFACE_HUB_CACHE|HF_ENDPOINT)=' | sort
python -c 'import os; names=("http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY","all_proxy","ALL_PROXY"); print("proxy variables set:", [name for name in names if os.environ.get(name)])'
```

The first command intentionally excludes `HF_TOKEN`; the proxy check prints names, not values.
Never print credentials while collecting that environment summary.

Install `.[evaluation]` only when implementing the ArcFace/Qwen evaluation stage; it adds
InsightFace, ONNX Runtime GPU, and Qwen vision utilities and is not required for the first SDXL
generation smoke test:

```bash
python -m pip install -e ".[gpu,evaluation,dev]"
```

If `check_model_assets.py` fails only for a cached asset, confirm the pinned snapshot path and the
`HF_HUB_CACHE`/`HF_HOME` values. Do not redownload everything automatically.

## Recommended GPU implementation order

1. Completed: run one authorized 768×768 prompt-only SDXL pipeline smoke test.
2. Completed: implement and smoke-test one global Canny ControlNet backend with injected-pipeline
   unit tests.
3. Implement DINOv2 Base and InsightFace pair metrics with explicit no-face/failure policies. Keep
   raw embeddings out of published artifacts unless their privacy implications are addressed.
4. Implement face parsing and region-aware Canny. Validate masks visually before batch generation.
5. Implement Qwen2.5-VL-3B style-removal scoring as a structured rubric, calibrate it against a
   human-labeled set, and use 7B only for robustness auditing.
6. Add DWPose/OpenPose. Depth, Refiner, InstantID, RealVisXL, SigLIP, and Florence remain extensions
   until the primary matrix is stable.

## Guardrails

- Never report copy backend or smoke similarity as a research result.
- A downloaded model is not a verified model; record real load and inference failures.
- Keep thresholds provisional until calibrated on human annotations.
- Do not commit models, raw faces, caches, embeddings, bulk outputs, or checkpoints.
- Review weight licenses independently from this repository's Apache-2.0 license.
- Florence custom code must be pinned and reviewed before enabling `trust_remote_code`.
- At the end of each experiment handoff, provide a `scripts/package_run.py --cleanup` command for
  the exact run directory. Download and verify the resulting ZIP and `.zip.sha256`; never delete
  outputs broadly before a verified archive exists.

## Prompt-only GPU verification completed

On 2026-08-14, commit `773d6ce` loaded the pinned local SDXL snapshot and completed one adaptive
and one generic 768×768 run on an NVIDIA GeForce RTX 4080 SUPER vGPU. Both used seed 42, BF16,
28 configured steps, strength 0.45, guidance 3.5, and EulerDiscreteScheduler. The records reported
8,745,006,080 peak allocated bytes and 10,322,182,144 peak reserved bytes. Recorded inference time
was 3.49 seconds for adaptive and 5.45 seconds for generic; these timings describe that instance
only and are not method-quality evidence.

Visual inspection found that both outputs retained strong oil-painting texture. The source itself
visually appears to be an oil painting but was invoked with `--style-category comic`, so this run
cannot evaluate the adaptive comic prompt. The prompt variants produced visibly different pixels,
but this is only a working pipeline smoke test, not a positive or negative research result. Use an
accepted sample whose declared category matches the image for the first meaningful comparison.

The next implementation is global Canny ControlNet. Its first GPU run must use the same authorized
source, adaptive prompt, seed, resolution, steps, strength, and guidance as the prompt-only run, and
must inspect the saved `.canny.png` condition before interpreting the generated image. Region-aware
Canny remains a later stage.

## Global Canny GPU verification completed

Later on 2026-08-14, commit `fe7ace3` loaded both pinned local snapshots and completed the global
Canny run on the same RTX 4080 SUPER vGPU. The structured record reported 15.05 seconds to load the
pipeline, 5.52 seconds for the measured inference section, 11,140,951,040 peak allocated bytes, and
11,383,341,056 peak reserved bytes. These are single-instance diagnostics, not comparative claims.

The saved condition was a valid binary 768×768 Canny image using thresholds 90/190 and conditioning
scale 0.8. Analysis of the returned condition copy found 15.16% nonzero edge pixels. It preserved
the main silhouette but also captured extensive oil-paint brush texture. The generated image still
looked like an oil painting and visibly redrew facial details, clothing, and background elements.
Because the oil-painting demo was still declared as `comic`, this verifies the backend and exposes a
plausible limitation of dense global edges, but it is not a valid method comparison. The next
meaningful run must use one accepted source whose declared style matches the image, with matched
prompt-only and global-Canny settings.

## First category-matched comic pilot

The accepted fictional source `synthetic-comic-001` was run through generic prompt-only, adaptive
prompt-only, and adaptive global Canny with seed 42. All three used the same SDXL revision,
768×768 size, EulerDiscreteScheduler, 28 configured steps, strength 0.45, guidance 3.5, BF16, and
the same style-specific negative prompt. The generic/adaptive factor therefore changed the positive
prompt only. Global Canny used thresholds 90/190 and conditioning scale 0.8.

All three outputs retained an unmistakable comic appearance, so this pilot does not support a
successful style-removal claim. Generic and adaptive were visually close. Global Canny preserved
the source more closely while visibly changing the prompt-only outputs, consistent with active
structural conditioning but not with improved destylization. Descriptive pixel checks against the
resized source gave MAE 15.16 for generic, 14.89 for adaptive, and 10.07 for Canny; generic versus
adaptive MAE was 4.08. The Canny condition had 9.91% nonzero pixels. These are diagnostic pixel
statistics, not DINO, ArcFace, VLM, or human-evaluation results.

Before expanding to a batch, use pilot data to find a viable img2img strength range; 0.45 appears
too conservative on this sample. Keep input, seed, model, scheduler, steps, guidance, and prompts
fixed while sweeping strength, and do not open calibration/test data for this choice. Recheck
global Canny only after prompt-only generation demonstrates meaningful style removal.

## Next pilot strength sweep

The private pilot inventory currently contains 20 QC-accepted sources: five each for `comic`,
`3d_cartoon`, `ink`, and `watercolor`. It is sufficient for the next exploratory sweep; no extra
demo upload is needed. `scripts/build_runtime_manifest.py` now converts the rich private pilot
manifest into a strict temporary runtime manifest while verifying transferred image checksums.
This does not freeze or publish the dataset.

Run adaptive prompt-only at strengths 0.50, 0.60, 0.70, and 0.80 over all 20 pilot sources with
seed 42. `scripts/run_strength_sweep.py` reuses one loaded SDXL pipeline and creates four isolated
run directories, for 80 outputs total. Hold guidance 3.5, 28 configured steps, resolution 768×768,
model revision, scheduler, and prompts fixed. This is parameter-range exploration on pilot data,
not a primary-method result. Inspect the four matched outputs per source and choose a viable range
before running Canny; do not generate the full Canny grid yet.
