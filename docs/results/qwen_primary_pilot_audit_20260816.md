# Qwen2.5-VL-3B primary-pilot rubric audit

This audit covers the completed 100-pair pilot evaluation: five methods by 20 source images. It
tests evaluator behavior, not method quality.

## Score compression

- Content: 98 records scored 4; the same historical source produced the only two scores of 2.
- Recoverable identity: 98 records scored 4; the same two records scored 3.
- Style removal: 80 records scored 3 and 20 scored 4.
- For every source image, all five methods received exactly the same style-removal score.
- Only two score triplets dominate: `(4, 3, 4)` and `(4, 4, 4)`, plus two `(2, 4, 3)` exceptions.

The style score therefore varies by source ID but not by generation method. Its identical 3.2
method mean is not a genuine finding that the five methods remove style equally well.

## Evidence behavior

The short evidence sometimes notices photographic conversion, watercolor/ink residue, cartoon
geometry, or identity drift. In other records it merely lists visible objects, clothing, or scene
content and does not justify the style score. The text can differ between methods while the numeric
style score remains fixed. This indicates that evidence sensitivity did not translate into useful
score discrimination under rubric version `face-destyle-paired-0to5-v1`.

## Decision

- Do not use the 3B style score for thresholding, method ranking, or routing on this pilot.
- Do not repair the problem by tuning on the held-out test set.
- Complete blinded human scoring first.
- Run Qwen2.5-VL-7B only on a predeclared disagreement subset after human scores are frozen.
- Any later rubric revision is a separately versioned evaluator experiment and must be calibrated
  on calibration sources rather than silently replacing these frozen raw records.
