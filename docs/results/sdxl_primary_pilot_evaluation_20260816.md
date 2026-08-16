# SDXL primary-pilot evaluation return audit

The returned `evaluation-sdxl-primary-pilot-20260816.zip` passed ZIP CRC verification. Its SHA-256
is `12829030046a422d2a3adc158c4037144cda1f920f33a3da661a55087f10c83b`.
The archive remains outside Git because it contains raw per-pair evaluator responses and server
paths. This document preserves the small, non-sensitive audit result.

The evaluation covers 80 pairs: four SDXL methods by the same 20 previously observed pilot sources.
It is not a held-out test.

## First-pass completeness

- DINOv2 Base: 80/80 scores.
- CLIP ViT-L/14: 0/80 scores. Transformers 5.15 returned a structured pooling object rather than
  the older tensor shape, exposing a compatibility bug.
- ArcFace: 79/80 cosine scores and one explicit `no_face_generated` result for
  `prompt_generic:met-12464`.
- Qwen2.5-VL-3B: 74/80 parsed rubric records. Six responses used integral floats or digit strings
  where the initial parser required JSON integers.

Every first-pass record contains a CLIP failure, so the returned file is an incomplete metric run,
not a formal result. The evaluator now accepts both CLIP return shapes and safely normalizes
integral Qwen score representations while retaining the frozen 0--5 bounds. Retry only CLIP and
failed Qwen entries; do not recompute successful DINO or ArcFace values.

## Preliminary diagnostic

Global Canny has the highest DINO and ArcFace means in every style, consistent with its already
observed tendency to retain source geometry and artistic contours. This is not evidence of better
style removal. Qwen showed little separation: most parsed records received content 4, identity 4,
and style removal 3. Blinded human calibration remains necessary, and the raw metrics must not be
silently averaged into an unexplained composite score.
