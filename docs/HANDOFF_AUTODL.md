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

## 2026-08-17 current execution boundary

The 120-source formal inventory is frozen in `data/manifests/formal-v1/inputs.jsonl` and resolves
against the common external `Face-DeStyle-Data` root. Calibration contains 40 sources (10 per
style); held-out test contains 60 (15 per style). Keep test sealed until pilot blind review,
calibration generation/annotation, thresholds, and routing rules are complete.

When GPU rental efficiency requires it, one frozen SDXL method may generate calibration and test in
the same model-loading session with repeated `--split calibration --split test`. Store the result as
one 100-record run, but do not inspect, score, select, or tune from its test subset before calibration
thresholds and routing rules are frozen. This compute optimization does not merge the split labels.

The FLUX stochastic extension is complete for seed 43 (20/20) and seed 44 (20/20). The returned
seed-44 ZIP passed CRC, record, and 1024x1024 RGB image validation; its SHA-256 is
`9c2042df4b4ee81074467b0cc2f395d044421f719e49a5cb3d84140d5fcce591`. Stop adding seeds. Do not
merge seeds 43/44 into the primary seed-42 evaluation or use them to tune prompts/settings. See
`docs/results/flux_seed_extension_20260817.md`.

Formal generation is complete: four SDXL methods each produced 100/100 calibration+test records,
while FLUX produced 40/40 calibration records. The five returned ZIPs and sidecars passed SHA-256,
ZIP CRC, frozen-manifest membership, unique-record/output-name, image decode, mode, and resolution
checks locally. The four SDXL archives contain 100 RGB 768x768 main outputs each; FLUX contains 40
RGB 1024x1024 main outputs. All use seed 42 and the frozen settings. Keep SDXL test outputs sealed
while calibration-only metrics and blinded human review are performed. Only the `3l8` AutoDL host
remains active; do not issue cross-host synchronization commands.

Returned formal archive SHA-256 values:

- prompt generic: `f790cbf4c27533f8fd391e3a5285e72794d7607d02fddd321fd09202234509c5`;
- prompt adaptive: `22a9f02c56f67e7573eb555822a0b8abfa464c292587f94527953490065a854a`;
- global Canny: `546908606c313c826e7d12f4022a43e1dfa6f841f0a5c0853ee67fb252692f45`;
- Region Canny: `19e91fb6e89430acf24b7fb393f3790bcc472e210b46ad5582cb077477ac743c`;
- FLUX calibration: `28c72e7f7d7ca5ca7927a9bfe0cce20e656f8c38bd08860c01ee0df1b51776bd`.

The calibration-only evaluation archive also passed SHA-256 and ZIP CRC validation locally. Its
SHA-256 is `3182e87dcbcedcf4d410ff679fce581d3e28b21c2a0264a53a0472b328a29ec2`.
It contains 200 records: DINO and CLIP completed 200/200; ArcFace produced 197 cosine values plus
three explicit `no_face_generated` statuses; no evaluator failure was recorded. ArcFace used the
CPU provider because this environment's ONNX Runtime did not expose CUDA. Do not rerun it merely
to change provider—the provider affects runtime, not the intended metric definition.

Calibration blind scoring is now frozen and unblinded. FLUX passed 37/40; the four SDXL methods
passed between 1/40 and 3/40, and SDXL fallback rescued none of the three FLUX failures. The next
and only formal-v1 generation task is the frozen 60-source FLUX test batch. Follow
`docs/HANDOFF_FORMAL_V1_HELDOUT_TEST.md` for the preregistered hypotheses, immutable settings,
machine validation, 300-pair blind review plus stratified 20% retest, paired statistics, packaging,
and stop conditions. Do not start another SDXL run or any LoRA/Multi-ControlNet extension.

## Formal-v1 held-out tooling prepared

The local-only held-out toolchain is now implemented and synthetic-tested before test GPU work:

- `scripts/build_heldout_blind_review.py` validates the exact five-method by 60-source test matrix,
  builds the 300-candidate primary round with seed 20260821, and selects/orders exactly three of 15
  candidates in every method-by-style cell for the 60-candidate repeat with seed 20260822. Method,
  source ID, filenames, generation metadata, and automatic metrics remain in the sealed private key.
- `scripts/validate_flux_test_archive.py` checks the ZIP sidecar, CRC, exact frozen test membership,
  unique records and outputs, prompts/settings, explicit failure records, and all 60 RGB 1024x1024
  decodes without displaying an image.
- `scripts/analyze_heldout_test.py` applies the frozen human pass rule and preregistered paired
  statistics, missing-data policy, per-style descriptions, metric-alignment diagnostics, and
  single-rater test-retest agreement. It does not select test thresholds or build a composite score.
- `scripts/run_flux_kontext_probe.py --resume` now validates existing success records, output files,
  failure records, paths, prompts, and frozen settings before skipping any completed source.

These tools do not authorize opening test images outside the method-hidden scoring materials. At
tooling freeze, the sole remaining server action was the frozen 60-source FLUX test generation on
`3l8`, followed by machine validation and packaging without `--cleanup`; its completion is recorded
below.

## 2026-08-21 formal-v1 test generation complete

The frozen 60-source FLUX test runner completed with exit code 0. Machine-only local inspection of
the returned ZIP found 60 unique frozen test records, 60 decodable RGB 1024x1024 outputs, consistent
BF16/offload/seed-42/28-step/guidance-2.5 settings, zero recorded failures, and a passing ZIP CRC.
No test image was displayed. The operator explicitly waives further hash and sidecar verification
for this specific archive and personally attests its provenance; report this as operator-attested,
not as a cryptographically verified transfer.

The exact 300-candidate primary blind round and 60-candidate stratified repeat are now built locally
with the frozen seeds. Reviewer-visible materials contain no method name, source ID, generation
path, or automatic metric, and the private key remains sealed. Do not run FLUX test generation
again. The next task is to complete and freeze the primary human scores, then the delayed repeat,
before unblinding or running held-out metrics/statistics. No LoRA, Multi-ControlNet, Qwen full pass,
or v2 experiment is authorized.

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

## Completed 20-source strength sweep

The adaptive prompt-only sweep completed on the RTX 4080 SUPER with 80/80 records and no recorded
generation failure. All records used the pinned SDXL revision, seed 42, Euler scheduler, 28
configured steps, guidance 3.5, 768×768, and BF16. The pipeline loaded once and was reused across
all four strengths. Mean recorded inference time per image ranged from 3.44 to 3.89 seconds; peak
allocated and reserved memory were 8,631,817,216 and 8,795,455,488 bytes. These timings are
instance diagnostics, not quality evidence.

Visual review found 0.50 and 0.60 generally too conservative. At 0.70, watercolor and ink sources
showed the clearest movement toward photographic rendering before frequent large reconstruction,
while comic and 3D-cartoon sources still retained substantial stylization. Strength 0.80 produced
more photographic texture in some outputs but frequently changed identity-bearing facial features,
age or gender presentation, accessories, hair, pose, and background details. No strength succeeded
uniformly across all styles. Mean source/output pixel MAE increased monotonically in every category;
this only confirms increasing pixel change and is not a style-removal or identity metric.

Use 0.70 as the single preselected pilot comparison point, not as a claimed optimum. The next GPU
run should hold it fixed and add the missing matched groups: generic prompt-only and adaptive global
Canny over the same 20 sources and seed 42. This yields the first controlled 20-source comparison
of the three implemented methods. Do not tune on calibration or test data, and do not interpret
the outputs as formal results until the planned metrics and blinded human rubric are applied.

## Completed three-method pilot comparison

The matched generic prompt-only and adaptive global-Canny runs completed for all 20 pilot sources
at strength 0.70. Combined with the prior adaptive prompt-only outputs, all three groups used seed
42, the pinned SDXL revision, Euler scheduler, 28 configured steps, guidance 3.5, 768×768, and BF16.
Generic used one shared positive prompt; adaptive and Canny used four category prompts. All methods
used the same four category-specific negative prompts. Canny used conditioning scale 0.8 and the
declared per-category thresholds.

Mean recorded inference time was 4.14 seconds for generic, 3.72 seconds for adaptive, and 5.52
seconds for Canny on the RTX 4080 SUPER. Prompt-only peak allocated memory was 8,631,817,216 bytes;
Canny peak allocated memory was 11,140,951,040 bytes. These are operational diagnostics only.

Visual review did not find a consistent adaptive-prompt advantage over generic. Their outputs were
usually close, and both still retained substantial comic or 3D-cartoon appearance. Global Canny
more consistently retained facial geometry, pose, composition, and small scene details, but it also
retained the source's drawn, inked, or painted contours. On several ink examples its output was
nearly a reconstruction of the source artwork rather than a destylized photograph. Source/output
pixel MAE was lower for Canny in every category, which is consistent with stronger source retention
but is not evidence of better quality or identity.

The scale-0.8 global condition is therefore too restrictive for this pilot. Before committing to a
region-aware implementation, keep strength 0.70 and all other settings fixed and test global-Canny
conditioning scales 0.4 and 0.6 over the same 20 sources. Compare those with the existing 0.8 group
to determine whether weaker global control yields a useful structure/style trade-off. This remains
pilot tuning, not a formal RQ1 or RQ2 result.

## Completed global-Canny scale sweep

The global-Canny scale-0.4 and scale-0.6 runs completed for all 20 pilot sources, with the prior
scale-0.8 group serving as the third matched level. All non-scale settings matched, and the 20 saved
condition images were byte-identical across levels. Mean recorded inference times were 5.72, 5.39,
and 5.52 seconds per image for scales 0.4, 0.6, and 0.8. Scale 0.4 had an unusually slow 46.69-second
pipeline load on that invocation; this isolated load time is not a method-quality result.

Increasing scale monotonically reduced source/output pixel MAE in every category, confirming
stronger source retention. Visual review found that 0.4 allowed more reconstruction than 0.6 or
0.8 while retaining useful structural control, so 0.4 is the selected global-Canny pilot setting.
However, all three levels still retained substantial artistic contours on comic, ink, and
watercolor sources. Further global-scale tuning is unlikely to address that failure mode.

The repository now includes a locally tested `region_canny` backend. It loads the registered
SegFormer face-parsing snapshot only on GPU and saves the global Canny image, parsed mask, and final
composite condition. The global and region ControlNet scale is preselected at 0.4 and img2img
strength at 0.70. This implementation state alone is not a GPU or quality result.

The first region-Canny smoke attempt stopped before segmentation because Transformers 5.15 could
not auto-detect the image processor from the pinned model's older preprocessor metadata. This was
an API-compatibility failure, not an OOM, inference, or mask-quality result. The backend now uses
`SegformerImageProcessor` explicitly while keeping the snapshot local and unchanged. The repeated
smoke after that fix is recorded below.

The repeated one-source smoke then completed on an RTX 4080 SUPER. The parser loaded from the pinned
local snapshot, the parsed head occupied 52.05% of this close-up source, and visual inspection found
that the mask excluded the background and most clothing. The composite condition also attenuated
the intended background edges. The generated image nevertheless remained unmistakably illustrated
and was close to the global-Canny result; this is a successful runtime/control-path smoke, not a
quality improvement or a method comparison. The fixed region representation keeps parsed
face/hair/neck edges at full strength and attenuates background edges to 0.25. Do not tune this
representation from the one smoke output. The next GPU task is one fixed-config Region Canny pass
over all 20 pilot sources, paired against the existing adaptive prompt-only and scale-0.4 global
Canny results. Hold strength 0.70, ControlNet scale 0.40, seed, prompts, scheduler, steps, guidance,
resolution, and model revisions fixed; do not add another parameter sweep.

## Completed 20-source Region Canny pilot

The fixed Region Canny pilot completed 20/20 records at commit `dc639ba` on an NVIDIA GeForce RTX
4080 SUPER with 32,760 MiB reported memory. The run used the same 20 accepted pilot sources as the
matched adaptive prompt-only and scale-0.4 global-Canny groups: five each from `comic`,
`3d_cartoon`, `ink`, and `watercolor`. All records used seed 42, the pinned SDXL, Canny ControlNet,
and face-parsing revisions, EulerDiscreteScheduler, 28 configured steps, strength 0.70, guidance
3.5, 768×768 BF16 generation, ControlNet scale 0.40, background edge scale 0.25, and the declared
per-style Canny thresholds. No controlled comparison field differed across the three method groups.

Mean Region Canny inference time was 5.68 seconds per image (range 4.86–8.16 seconds). The pipeline
and face parser loaded once in 13.42 and 1.82 seconds. Maximum recorded allocated and reserved GPU
memory were 11,668,694,528 and 13,505,658,880 bytes. These values describe this instance and are not
method-quality evidence.

All 20 global Canny images, parsed masks, region conditions, and outputs decoded at 768×768. Each
saved region condition was exactly reproducible as full global-Canny intensity inside the parsed
head mask and 0.25 intensity outside it. Mean parsed-mask fraction was 36.82%, ranging from 15.07%
to 52.05%. Visual inspection found broadly plausible masks on the synthetic portraits, but clear
semantic over-segmentation on `met-459963` and `aic-18667`, where hat, torso, furniture, or stray
regions entered the full-strength area; `met-12464` was also irregular. File validity and exact
compositing therefore verify implementation, not parsing quality across stylized domains.

Unblinded paired contact-sheet review did not find a stable Region Canny advantage. The 3D-cartoon
outputs remained visibly 3D, comic outputs remained illustrated, and ink/watercolor outputs retained
their source media. Background attenuation caused local reconstruction but did not consistently
remove artistic contours from the subject. As a software diagnostic only, Region and global-Canny
outputs were pixel-closer than Region and adaptive outputs on 16/20 sources; none of the Region and
global outputs were byte-identical. Pixel distance is not a content, identity, or style-removal
metric.

This pilot does not support H3 as currently represented. It suggests two limitations to analyze:
the face parser is not reliably semantic on every stylized or historical source, and keeping parsed
head edges at full strength can preserve the same artistic contours that global Canny preserved.
Do not tune weights from selected examples or run another Region parameter sweep. Freeze these
outputs, conduct the written paired human rubric, and validate the planned content, identity, and
style-removal evaluation paths before deciding whether a predeclared failure-aware route is warranted.

## Completed FLUX handoff

Further SDXL Base tuning remains stopped. The exploratory original-BF16 FLUX.1-Kontext-dev probe
completed 20/20 frozen pilot sources at native 1024x1024 with no Canny or parameter sweep. The run
used model offload, 28 steps, guidance 2.5, seed 42, and one pipeline-loading session. It recorded
zero failures, 1849.47 seconds total inference time, 92.47 seconds mean time per image, and
24,910,057,472 bytes peak allocated VRAM on an RTX 4080 SUPER.

The returned archive passed CRC and image validation; its SHA-256 is
`163e5e6517e8d48fd280a35f4cd41db2862d37b9a76133ea444080156a6961f5`. The operator intentionally
used configuration-file hashes instead of a full 54 GB weight SHA-256, and the manifest records
`large_weight_sha256=not_run_operator_choice`. Successful BF16 pipeline load and complete inference
serve as operational integrity evidence, not full cryptographic verification of all weight shards.

Unblinded review found a useful generator-capability signal for comic, ink, and watercolor but
persistent rendered geometry/materials for all five 3D-cartoon sources. This is not formal or
resolution-controlled evaluation: FLUX used native 1024 while the SDXL pilot used 768. Freeze the
outputs and proceed to blinded rubric calibration and failure-aware DINO/CLIP evaluation. Optional
paired ArcFace drift diagnostics are allowed only within the narrow private-research boundary in
`AGENTS.md`. Full run details and stop conditions are in `docs/HANDOFF_FLUX_KONTEXT.md`.

## Primary evaluation runner prepared

`scripts/evaluate_formal.py` now performs checkpointed, local-only DINOv2 Base, CLIP ViT-L/14,
paired ArcFace, and structured Qwen2.5-VL-3B evaluation across multiple run-record files while
loading each model only once. It stores raw cosine/rubric values and explicit failures; it does not
apply the placeholder thresholds. `scripts/summarize_formal_evaluations.py` writes factual CSV/JSON
summaries without declaring acceptance. Exact preflight, one-session evaluation, resume, packaging,
and subsequent experiment commands are in `docs/HANDOFF_EVALUATION.md`.

## Primary pilot evaluation completed

The compatibility-fixed run completed all raw fields for 80 SDXL and 20 FLUX pairs. DINO, CLIP,
and Qwen have 100/100 values. ArcFace has 99 cosine values plus one explicit
`prompt_generic:met-12464` no-face result; there are zero evaluator failures. The verified archive
SHA-256 values are `3986a0c570728e7997df77e799fb42bdd9fc4a046e07a250fe5ebf261d027e42`
for SDXL and `a061880253c883b3d9f84e7ade1fff9923a4a81f27ef35208edcedb6a952df07`
for FLUX.

Global Canny has the highest DINO, CLIP, and ArcFace means, which agrees with strong source/contour
retention and is not a style-removal result. Qwen assigns every method the same 3.2 mean
style-removal score and is not discriminative on this pilot. Freeze these raw values. Do not apply
placeholder thresholds or claim a winner; proceed to blinded human calibration and independent
calibration/test sources. The factual aggregate is in
`docs/results/sdxl_primary_pilot_evaluation_20260816.md`.
