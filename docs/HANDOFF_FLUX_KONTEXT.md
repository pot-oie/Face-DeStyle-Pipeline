# FLUX.1 Kontext generator-probe handoff

## Management sentence

> This experiment probes FLUX.1-Kontext-dev prompt editing on the same frozen sources as the SDXL
> Base pilot, using each model's recorded operating resolution, to test whether a stronger native
> image-editing generator improves natural reconstruction without unacceptable content drift.

## Research position and boundary

The SDXL Base pilot is frozen. Generic versus adaptive prompting did not show a stable difference;
global Canny retained artistic contours; and region-aware Canny completed 20/20 records without a
stable visual advantage. Do not run another SDXL Base prompt, strength, or Canny-weight scan.

The completed method ID is `flux1_kontext_dev_prompt_edit_bf16_offloaded`. Kontext natively consumes a
source image and editing instruction, so “prompt edit” here means image-conditioned editing without
an explicit Canny, Depth, or pose condition. This is a generator-capability extension, not a strict
reproduction of the DeStyle paper. The 20-output run is engineering and capability evidence, not a
formal result.

## Verified ModelScope source

- ModelScope mirror: `black-forest-labs/FLUX.1-Kontext-dev`
- Mirror revision: `master` (mutable; record retrieval time and hashes)
- Official identity: `black-forest-labs/FLUX.1-Kontext-dev`
- Reported repository size: 57,890,837,493 bytes
- Libraries/formats: Diffusers, PyTorch, Safetensors
- Reported tensor types: BF16 and F32
- License: FLUX.1 dev Non-Commercial License; the mirror does not relicense the model

Do not substitute `MusePublic/FLUX.1-Kontext-Dev`; ModelScope reports that repository as FP8 E4M3.
Also reject FP8, GGUF, NF4, NVFP4, ComfyUI-only, LoRA, and merged variants.

The BFL license requires project-specific interpretation. The operator has chosen a narrow
private-research interpretation that permits ArcFace/InsightFace only as a paired drift diagnostic
on these fixed inputs and outputs. Do not use it for identification, search, enrollment,
authentication, surveillance, or a real-person recognition system, and do not publish reusable face
templates. Record this interpretation with results; it is not legal advice or a claim that the
license wording is unambiguous.

## AutoDL download and verification

The download can run without a GPU. Keep it on the persistent data disk. ModelScope direct access
normally does not need AutoDL's academic proxy; the proxy is still required before pulling GitHub.

```bash
download_flux_kontext_modelscope() {
  set +e

  conda activate face-destyle
  unset OMP_NUM_THREADS

  export FACE_DESTYLE_ROOT=/root/autodl-tmp/face-destyle
  export KONTEXT_MODEL_ID=black-forest-labs/FLUX.1-Kontext-dev
  export KONTEXT_MODELSCOPE_REVISION=master
  export KONTEXT_DIR="$FACE_DESTYLE_ROOT/models/diffusion/FLUX.1-Kontext-dev"
  export KONTEXT_MANIFEST_DIR="$FACE_DESTYLE_ROOT/models/download-manifests"

  cd "$FACE_DESTYLE_ROOT/code/Face-DeStyle-Pipeline" || return 1
  source /etc/network_turbo
  git pull --ff-only origin main || return 1

  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

  python -c 'import modelscope; print("modelscope", modelscope.__version__)' || {
    echo 'ModelScope is unavailable; install it in the existing download environment first.'
    return 1
  }

  mkdir -p "$KONTEXT_DIR" "$KONTEXT_MANIFEST_DIR"

  modelscope download \
    --model "$KONTEXT_MODEL_ID" \
    --revision "$KONTEXT_MODELSCOPE_REVISION" \
    --local_dir "$KONTEXT_DIR"
  download_rc=$?

  if [ "$download_rc" -ne 0 ]; then
    echo 'Download stopped; keep the directory and rerun this function to resume.'
    return "$download_rc"
  fi

  for component in scheduler text_encoder text_encoder_2 tokenizer tokenizer_2 transformer vae
  do
    test -d "$KONTEXT_DIR/$component" || {
      echo "Missing Diffusers component: $component"
      return 1
    }
  done

  test -f "$KONTEXT_DIR/model_index.json" || {
    echo 'Missing model_index.json'
    return 1
  }

  export EXPECTED_HASHES="$KONTEXT_MANIFEST_DIR/flux1-kontext-dev-modelscope-master.sha256"
  printf '%s\n' \
    'afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38  ae.safetensors' \
    '843a26dc765d3105dba081c30bce7b14c65b0988f9e8d14e9fbc8856a6deebd5  flux1-kontext-dev.safetensors' \
    '893d67a23f4693ed42cdab4cbad7fe3e727cf59609c40da28a46b5470f9ed082  text_encoder/model.safetensors' \
    'ec87bffd1923e8b2774a6d240c922a41f6143081d52cf83b8fe39e9d838c893e  text_encoder_2/model-00001-of-00002.safetensors' \
    'a5640855b301fcdbceddfa90ae8066cd9414aff020552a201a255ecf2059da00  text_encoder_2/model-00002-of-00002.safetensors' \
    '5414f9ba3d3945512769b1b5ecd41122c8bcfebb7ec906a3ac60daf371d38946  transformer/diffusion_pytorch_model-00001-of-00003.safetensors' \
    'a6fa67a12833c30040f794365a187cac8eef2e467152b84319034ae2828d9d03  transformer/diffusion_pytorch_model-00002-of-00003.safetensors' \
    '464f5343b6b06a2337d8c34b72ebec38219ebbfb7aafecbc9b342dc031417e7a  transformer/diffusion_pytorch_model-00003-of-00003.safetensors' \
    'f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3  vae/diffusion_pytorch_model.safetensors' \
    > "$EXPECTED_HASHES"

  cd "$KONTEXT_DIR" || return 1
  sha256sum -c "$EXPECTED_HASHES" || return 1

  incomplete_count=$(find "$KONTEXT_DIR" -type f \
    \( -name '*.incomplete' -o -name '*.aria2' -o -name '*.tmp' \) \
    | wc -l | tr -d ' ')
  if [ "$incomplete_count" != '0' ]; then
    echo "Found $incomplete_count incomplete files"
    return 1
  fi

  export KONTEXT_FULL_HASHES="$KONTEXT_MANIFEST_DIR/flux1-kontext-dev-modelscope-master.all-files.sha256"
  find . -type f -print0 | sort -z | xargs -0 sha256sum > "$KONTEXT_FULL_HASHES"

  export KONTEXT_DOWNLOAD_MANIFEST="$KONTEXT_MANIFEST_DIR/flux1-kontext-dev-modelscope-master.txt"
  {
    echo "recorded_at=$(date -Is)"
    echo "model_id=$KONTEXT_MODEL_ID"
    echo "official_model_id=black-forest-labs/FLUX.1-Kontext-dev"
    echo "source=modelscope_mirror"
    echo "source_revision=$KONTEXT_MODELSCOPE_REVISION"
    echo 'reported_repository_bytes=57890837493'
    echo 'reported_tensor_types=BF16,F32'
    echo 'license=FLUX.1-dev-Non-Commercial-License-v1.1.1'
    echo "local_path=$KONTEXT_DIR"
    echo "expected_hash_manifest=$EXPECTED_HASHES"
    echo "full_file_hash_manifest=$KONTEXT_FULL_HASHES"
    echo 'large_file_sha256=passed'
    echo "resolved_size=$(du -sh "$KONTEXT_DIR" | awk '{print $1}')"
  } | tee "$KONTEXT_DOWNLOAD_MANIFEST"

  echo 'KONTEXT_MODELSCOPE_DOWNLOAD_COMPLETED=1'
  echo "KONTEXT_DIR=$KONTEXT_DIR"
  echo "KONTEXT_MODELSCOPE_REVISION=$KONTEXT_MODELSCOPE_REVISION"
  echo "KONTEXT_EXPECTED_HASHES=$EXPECTED_HASHES"
  echo "KONTEXT_FULL_HASHES=$KONTEXT_FULL_HASHES"
  echo "KONTEXT_DOWNLOAD_MANIFEST=$KONTEXT_DOWNLOAD_MANIFEST"
}

download_flux_kontext_modelscope
```

If `modelscope` is absent, do not silently change the `face-destyle` inference environment. Use the
existing ModelScope download environment from the Qwen acquisition or create a separate downloader
environment, then rerun the same command. An interrupted download is resumable; do not delete a
partial directory merely because one invocation stopped.

## Implementation contract for the new task

Do all implementation locally with injected mocks before using a GPU. Add the downloaded asset to
the registry only after the returned directory and hash manifest are known. The first backend must:

- use `FluxKontextPipeline` from the local Diffusers-format directory only;
- use BF16, batch size 1, native 1024×1024, and CPU/model offload;
- keep text encoders on CPU or release/offload them before denoising;
- record the exact ModelScope source, mutable revision label, file hashes, local path, package
  versions, prompt, seed, steps, guidance, runtime, RAM, and peak VRAM;
- identify itself as `flux1_kontext_dev_prompt_edit_bf16_offloaded`;
- preserve every failure and never overwrite a previous output;
- run the repository's Ruff, pytest, all script-help, and diff checks before handoff.

The initial manifest must deterministically select one accepted pilot source from each of
`comic`, `3d_cartoon`, `ink`, and `watercolor`. Do not select examples after seeing FLUX output.
Cross-model seed equality is not matched noise; compare paired sources, resolution, and task intent,
while reporting each generator's own fixed configuration.

## GPU sequence and stop conditions

1. Check GPU availability, GPU memory, system RAM, and swap.
2. Run one of the four preselected sources first.
3. If BF16 model offload completes, the operator may run the complete frozen 20-source pilot in one
   pipeline-loading session because GPU rental time is limited. This is an opportunistic exploratory
   batch, not a formal cross-model comparison.
4. Keep every pilot output; do not select examples after generation.
5. On OOM, record whether it occurred during loading, text encoding, denoising, or VAE decode.
6. Only a documented BF16/offload OOM may justify a separately named quantized/offloaded probe.
7. Do not add FLUX Canny, Depth, Pose, LoRA, or a parameter sweep.

Package a completed run with `scripts/package_run.py --cleanup` only after ZIP and SHA-256
verification. The handoff must include every attempted source and output in non-cherry-picked
contact sheets, success/failure counts, timing, peak VRAM, content drift, residual style, and what
the probe can and cannot establish.

## Completed native-1024 pilot

Run `flux-kontext-native1024-pilot-20260815-023158` completed all 20 frozen sources with zero
failures on an RTX 4080 SUPER. It used BF16, CPU/model offload, 28 steps, guidance 2.5, seed 42,
native 1024x1024 output, and one pipeline load. Pipeline loading took 21.07 seconds; total recorded
inference took 1849.47 seconds (mean 92.47, range 87.65--116.40 seconds per image). Peak allocated
VRAM was 24,910,057,472 bytes and peak reserved VRAM was 26,216,497,152 bytes. The returned ZIP
passed CRC testing and has SHA-256
`163e5e6517e8d48fd280a35f4cd41db2862d37b9a76133ea444080156a6961f5`.

The operator deliberately skipped hashing every 54 GB weight shard. The acquisition manifest
records lightweight configuration hashes and `large_weight_sha256=not_run_operator_choice`; a
successful local BF16 pipeline load and complete inference run are the operational integrity check.
Do not retroactively describe this as full weight verification.

An unblinded, uncalibrated contact-sheet review found strong style-removal signal on comic and
watercolor, mixed results on ink, and residual 3D/cartoon geometry and materials on all five
3D-cartoon sources. Historical artworks also exposed identity-inference uncertainty. Freeze the
outputs: do not tune from selected examples. The next stage is blinded human-rubric calibration and
failure-aware DINO/CLIP evaluation, with optional paired ArcFace drift diagnostics under the narrow
operator interpretation above.

## Local probe runner prepared

The repository now has a locally mock-tested `FluxKontextBackend` and
`scripts/run_flux_kontext_probe.py`. The runner checksum-validates the portable pilot manifest,
selects records in a deterministic style/source order, refuses output overwrite, loads only the
supplied local Diffusers directory, uses native 1024×1024 BF16 model CPU offload, and appends one
success or failure record after every attempted source. `--probe-stage pilot` runs the complete
frozen pilot with one pipeline load. `--resume` skips successful source IDs already present in the
records JSONL after an interrupted rental session.

Do not add the model to `configs/models.yaml` until the server download manifest and hash manifest
paths have been confirmed. Pass those verified paths directly to the probe runner in the meantime.
