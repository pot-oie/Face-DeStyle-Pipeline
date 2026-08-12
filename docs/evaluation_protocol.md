# Evaluation protocol

Build a held-out, human-annotated calibration set before choosing thresholds. Freeze its split and
document inter-annotator instructions. Tune `configs/evaluation.yaml` only on calibration data, then
evaluate once on the held-out test split.

Planned dimensions are DINO/CLIP content preservation, ArcFace identity preservation, VLM-based
style removal, and blinded human acceptance rate. Report distributions and uncertainty, not only
means. Break down outcomes by style, prompt mode, control mode, demographic strata where ethically
and statistically appropriate, and failure type.

The current `smoke_test_similarity` is normalized pixel similarity for software verification. Its
style score is an explicitly labeled sentinel. Neither value is valid evidence of destylization.
The thresholds shipped in configuration are placeholders and require calibration.
