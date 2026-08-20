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

The completed 100-pair pilot exposed severe Qwen2.5-VL-3B score compression: every method received
the same style-removal score for a given source, even when its evidence and human visual impression
differed. Freeze those raw records, do not use their style score for acceptance, and complete the
method-hidden human review first. Qwen 7B is limited to the frozen human/model disagreement subset;
it is not an automatic replacement evaluator over the whole pilot.

Use `scripts/build_blind_review.py` to create two deterministically shuffled review rounds with
opaque IDs and equal-size source/candidate panels. One complete round may be frozen as the primary
calibration annotation when reviewer burden prevents a repeat; decide this before unblinding and
record that within-reviewer agreement is unavailable. A completed second round is otherwise used
only to estimate repeat agreement. Keep `private/private_key.jsonl` closed until the included rounds
are frozen, then use `scripts/summarize_blind_review.py`. For the formal face set, the pass rule is
content >= 4, style removal >= 4, identity judgment valid, and identity >= 4; an output with no
judgeable face is rejected rather than exempted from the identity gate. Blank failure-type fields
mean “not reported,” not “no failure”; failure subtypes are secondary diagnostics and do not
determine the main pass rate. Report score distributions, pass rates, available agreement, and
failure-label coverage rather than only means.
