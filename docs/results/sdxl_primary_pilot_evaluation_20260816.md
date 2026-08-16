# Primary-pilot evaluation return audit

Two completed evaluation archives passed their embedded SHA-256 checks and ZIP CRC verification:

- SDXL 80-pair archive: `3986a0c570728e7997df77e799fb42bdd9fc4a046e07a250fe5ebf261d027e42`;
- FLUX 20-pair archive: `a061880253c883b3d9f84e7ade1fff9923a4a81f27ef35208edcedb6a952df07`.

The archives remain outside Git because they contain raw per-pair evaluator responses and server
paths. This document preserves the small, non-sensitive audit result. All 100 pairs use the same
previously observed 20-source pilot, not held-out test data. FLUX also used native 1024x1024 while
SDXL used 768x768, so cross-generator values are exploratory rather than a controlled resolution
comparison.

## Completeness

- DINOv2 Base: 100/100 scores.
- CLIP ViT-L/14: 100/100 scores.
- ArcFace: 99/100 cosine scores. `prompt_generic:met-12464` explicitly records
  `no_face_generated`; every FLUX pair produced a valid largest-face result.
- Qwen2.5-VL-3B: 100/100 parsed rubric records.
- Records with an evaluator failure: 0.

The first SDXL pass exposed a Transformers 5.15 CLIP return-shape incompatibility and six Qwen
integral-value representation differences. The compatibility retry filled only the missing fields
and preserved successful DINO and ArcFace values.

## Raw method means

These are descriptive raw values, not calibrated acceptance probabilities.

| Method | DINO cosine | CLIP cosine | ArcFace cosine | Qwen content | Qwen style removal | Qwen identity |
|---|---:|---:|---:|---:|---:|---:|
| SDXL generic | 0.646 | 0.735 | 0.265 (19/20) | 3.9 | 3.2 | 3.95 |
| SDXL adaptive | 0.693 | 0.761 | 0.291 | 4.0 | 3.2 | 4.0 |
| SDXL global Canny 0.4 | 0.907 | 0.888 | 0.596 | 3.9 | 3.2 | 3.95 |
| SDXL Region Canny | 0.779 | 0.839 | 0.566 | 4.0 | 3.2 | 4.0 |
| FLUX Kontext native 1024 | 0.735 | 0.772 | 0.519 | 4.0 | 3.2 | 4.0 |

Global Canny leads all three cosine means, consistent with its already observed tendency to retain
source geometry and artistic contours. That is content/structure retention, not evidence of better
style removal. Region Canny reduces this retention relative to global Canny but still does not show
a measured style-removal advantage.

Qwen gives every method the same mean style-removal score of 3.2 and nearly constant content and
identity scores. It therefore does not distinguish the observed qualitative differences in this
pilot. The earlier unblinded FLUX review found stronger photographic conversion for comic, ink, and
watercolor but persistent 3D rendering cues; the automated rubric does not reproduce that pattern.
Blinded human calibration is required before thresholds, acceptance, routing, or method ranking.
