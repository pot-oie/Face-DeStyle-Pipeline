# Formal-v1 calibration source-similarity metrics — 2026-08-20

## Scope and integrity

This is a calibration-only result, not a held-out test result. It covers 40 frozen calibration
sources, balanced ten each across `3d_cartoon`, `comic`, `ink`, and `watercolor`, for five methods
and 200 matched method-source pairs. The selection manifest contains calibration records only. The
returned archive passed its SHA-256 sidecar and ZIP CRC; its SHA-256 is
`3182e87dcbcedcf4d410ff679fce581d3e28b21c2a0264a53a0472b328a29ec2`.

DINOv2 Base and CLIP ViT-L/14 completed 200/200 pairs. Paired ArcFace produced 197 cosine values and
three explicit `no_face_generated` statuses: `prompt_generic:met-415530`,
`region_canny:met-335049`, and `region_canny:met-339919`. There were no evaluator failures. ONNX
Runtime exposed only its CPU provider, so ArcFace ran on CPU; this is a runtime fact, not a change
to the metric definition.

## Raw method aggregates

| Method | DINO mean | CLIP mean | ArcFace mean | ArcFace n |
|---|---:|---:|---:|---:|
| Prompt generic | 0.5393 | 0.6797 | 0.1966 | 39 |
| Prompt adaptive | 0.5849 | 0.7014 | 0.2143 | 40 |
| Global Canny 0.4 | 0.8829 | 0.8934 | 0.5240 | 40 |
| Region Canny | 0.7311 | 0.8008 | 0.4546 | 38 |
| FLUX Kontext native 1024 | 0.7509 | 0.7535 | 0.4326 | 40 |

These are source/output similarity measures. They do not measure photographic realism or style
removal and must not be combined into an unexplained total score.

## Paired calibration contrasts

The intervals below are nonparametric 95% bootstrap confidence intervals over matched calibration
sources using 20,000 resamples and fixed analysis seed 20260820. ArcFace uses complete pairs only.

| Contrast (left minus right) | Metric | Mean difference | 95% bootstrap CI | Left wins |
|---|---|---:|---:|---:|
| Adaptive − generic | DINO | +0.0456 | [+0.0041, +0.0919] | 22/40 |
| Adaptive − generic | CLIP | +0.0218 | [+0.0033, +0.0404] | 27/40 |
| Adaptive − generic | ArcFace | +0.0209 | [−0.0011, +0.0434] | 24/39 |
| Global Canny − adaptive | DINO | +0.2980 | [+0.2472, +0.3504] | 40/40 |
| Global Canny − adaptive | CLIP | +0.1919 | [+0.1572, +0.2290] | 40/40 |
| Global Canny − adaptive | ArcFace | +0.3097 | [+0.2773, +0.3421] | 40/40 |
| Region Canny − adaptive | DINO | +0.1463 | [+0.1011, +0.1955] | 38/40 |
| Region Canny − adaptive | CLIP | +0.0994 | [+0.0754, +0.1248] | 38/40 |
| Region Canny − adaptive | ArcFace | +0.2290 | [+0.1887, +0.2677] | 36/38 |
| Region Canny − global Canny | DINO | −0.1517 | [−0.1960, −0.1119] | 0/40 |
| Region Canny − global Canny | CLIP | −0.0925 | [−0.1185, −0.0684] | 1/40 |
| Region Canny − global Canny | ArcFace | −0.0781 | [−0.1122, −0.0494] | 4/38 |
| FLUX − adaptive | DINO | +0.1660 | [+0.1057, +0.2260] | 37/40 |
| FLUX − adaptive | CLIP | +0.0520 | [+0.0176, +0.0863] | 27/40 |
| FLUX − adaptive | ArcFace | +0.2182 | [+0.1689, +0.2686] | 36/40 |

## Interpretation boundary

Adaptive prompting shows a small calibration-set source-similarity increase over the generic
prompt, but the ArcFace interval includes zero and none of these metrics establishes stronger style
removal. Global Canny maximizes source retention, consistent with preserving contours as well as
desired structure. Region Canny relaxes that retention relative to Global Canny while retaining
more source information than prompt-only. Whether this is a better content/style trade-off cannot
be decided until blinded style-removal and content labels exist.

FLUX retains more source information than adaptive SDXL prompt-only on these metrics, but it uses a
different architecture and native 1024 resolution. Its very high 3D-cartoon similarity and much
lower ink/watercolor similarity are compatible with the earlier visual observation of
style-dependent behavior; they are not evidence that the output is more photographic.

The next action is one method-hidden calibration review using the written 0–5 content,
style-removal, and recoverable-identity rubric. Freeze human-derived acceptance and routing rules
before opening any test output. Do not select a method from these similarity values alone, rerun
Qwen2.5-VL-3B broadly, or add another generation sweep.
