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

This is a file-integrity result, not a GPU inference result. Canny, pose, depth, refiner, InstantID,
RealVisXL, DINO/CLIP/SigLIP, ArcFace, Florence, and Qwen metric execution remain unverified.

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

Install `.[evaluation]` only when implementing the ArcFace/Qwen evaluation stage; it adds
InsightFace, ONNX Runtime GPU, and Qwen vision utilities and is not required for the first SDXL
generation smoke test:

```bash
python -m pip install -e ".[gpu,evaluation,dev]"
```

If `check_model_assets.py` fails only for a cached asset, confirm the pinned snapshot path and the
`HF_HOME`/`HUGGINGFACE_HUB_CACHE` values. Do not redownload everything automatically.

## Recommended GPU implementation order

1. Run one authorized 768×768 prompt-only SDXL image with the existing backend and record peak
   allocated/reserved VRAM, runtime, package versions, scheduler, seed, and output metadata.
2. Implement one Canny ControlNet backend with injected-pipeline unit tests, then run the same image,
   seed, prompt, and generation settings.
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
