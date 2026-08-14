# Limitations and responsible use

Stylized faces may not retain enough evidence to recover a unique real appearance. Outputs are
model-dependent reconstructions, not ground truth. Identity embeddings can be demographically
biased; VLM judgments may confuse aesthetics with style removal; Canny can preserve stylized
contours; central masks are not face segmentation. The region-aware backend uses a
CelebAMask-HQ-derived parser whose masks can fail on stylized, occluded, or out-of-distribution
faces and can inherit training-data bias. Human review remains necessary.

Do not use this project for identification, surveillance, impersonation, or claims about a person's
true appearance. Obtain authorization for face data, minimize retention, secure raw data, honor
removal requests, and disclose synthetic transformations. Assess licensing separately for datasets,
models, and outputs.
