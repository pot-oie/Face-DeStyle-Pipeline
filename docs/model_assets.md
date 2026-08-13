# Model assets and experiment status

`configs/models.yaml` is the source-controlled registry for server-side model assets. It records
roles, expected files, upstream identifiers, available pinned revisions, loader types, and licenses.
It never downloads a model. Local paths are relative to `FACE_DESTYLE_ROOT`; cached assets resolve
through `HF_HOME` and a pinned Hugging Face snapshot.

The inventory supplied on 2026-08-14 reported 51.15 GiB under the AutoDL model directory and
11.37 GiB in the Hugging Face cache. All ten HFD manifests were complete, both Qwen model indexes
resolved every shard, all Safetensors headers validated, and no partial files or broken links were
reported. This validates file presence and container format only. It does not validate GPU loading,
numerical behavior, metric calibration, or scientific quality.

Registered primary assets are SDXL base, Canny ControlNet, OpenPose ControlNet, face parsing,
DWPose, DINOv2 Base, CLIP ViT-L/14, InsightFace buffalo_l, and Qwen2.5-VL-3B. Extension assets are
SDXL Refiner, the FP16-fix VAE, Depth ControlNet, Depth Anything V2 Large, InstantID, RealVisXL V5,
DINOv2 Large, SigLIP SO400M, Florence-2 Large, and Qwen2.5-VL-7B.

Run the offline presence audit on AutoDL with:

```bash
export FACE_DESTYLE_ROOT=/root/autodl-tmp/face-destyle
export HF_HOME="$FACE_DESTYLE_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
python scripts/check_model_assets.py --config configs/models.yaml
```

The following assets have non-Apache restrictions that must remain visible in reports and release
documentation: SDXL-family OpenRAIL licenses, InsightFace and face-parsing research restrictions,
Depth Anything V2 CC-BY-NC-4.0, Qwen Research License, and any upstream uncertainty recorded in the
registry. Repository Apache-2.0 does not relicense weights.

Florence-2 uses repository-provided Python modeling code. Pin and review that code before enabling
`trust_remote_code`; the asset audit deliberately does not import it. RealVisXL is a single-file
checkpoint and needs a separate `from_single_file` backend before it can be executed.
