# Formal-v1 calibration blinded human review — Round A

## Protocol status

One complete method-hidden calibration round was frozen before opening the private method key. The
frozen `scores.csv` SHA-256 is
`3e686ab296ee9c12f1832a9ce2fe0c4d0d3e4bccc57c24ccbc5de435e08ba213`.
It contains 200 unique blind IDs: 40 calibration sources, five methods, and 50 candidate pairs per
style. All content, style-removal, and identity fields contain valid integer scores from 0 to 5;
198 identity judgments are valid and two are explicitly invalid.

The operator stopped after Round A because one full calibration round was sufficient for this
undergraduate reproduction. This decision was made while method identities remained hidden. Round
B is waived and will not be used later; therefore no within-reviewer repeat-agreement statistic is
available. This is a documented precision limitation, not test leakage or method selection.

Only 54/200 rows contain a manual failure subtype. Another 146 are blank, including 101 with at
least one core score below 4. Blank failure fields are therefore treated as `not_reported`, not as
evidence that no failure occurred. The complete core scores determine acceptance; the incomplete
failure labels are descriptive examples only.

## Frozen acceptance rule

A formal face candidate passes only when:

- content preservation is at least 4;
- style removal is at least 4;
- facial identity is judgeable; and
- recoverable facial identity is at least 4.

An unjudgeable face is rejected rather than exempted from the identity gate. This clarification has
no effect on the observed pass counts because both unjudgeable candidates also failed content and
style removal.

## Main result

| Method | Content mean | Style-removal mean | Identity mean | Passed | Pass rate |
|---|---:|---:|---:|---:|---:|
| FLUX Kontext native 1024 | 4.350 | 3.975 | 4.325 | 37/40 | 92.5% |
| Global Canny 0.4 | 3.300 | 2.550 | 3.325 | 3/40 | 7.5% |
| Region Canny | 3.175 | 2.600 | 3.100 | 3/40 | 7.5% |
| Prompt generic | 2.850 | 2.550 | 2.625 | 2/40 | 5.0% |
| Prompt adaptive | 2.950 | 2.375 | 2.650 | 1/40 | 2.5% |

FLUX passed 8/10 `3d_cartoon`, 9/10 `comic`, 10/10 `ink`, and 10/10 `watercolor`
candidates. Its three failures were `synthetic-3d-cartoon-009` (3/3/3),
`synthetic-3d-cartoon-011` (4/3/4), and `synthetic-comic-009` (4/4/3) for
content/style/identity. No SDXL alternative passed for any of these three sources, so the frozen
SDXL fallback set adds zero accepted calibration sources after FLUX.

## Interpretation and frozen next-stage rule

The human result reverses the apparent ordering of the source-similarity metrics: Global Canny had
the highest DINO, CLIP, and ArcFace similarity but very low human style-removal and acceptance.
This confirms that source retention cannot serve as a style-removal score. Region Canny does not
produce a useful human-acceptance advantage over Global Canny, and adaptive prompting does not
improve acceptance over the generic prompt on this calibration set.

For the held-out comparison, freeze FLUX as the primary method. Apply the same human rule once on
test. If FLUX fails, reject the source; do not route it to the existing SDXL candidates, because
that fallback rescued 0/3 FLUX calibration failures. Retain the four SDXL methods as formal
comparison baselines and report their outcomes, but do not use them as a progressive rescue route.
This is a negative calibration result for the proposed RQ3 routing strategy, not a claim that every
possible future fallback model would fail.

The strong FLUX calibration result supports generating its frozen 60-source test split at native
1024 with the same seed and settings. The numerical FLUX expansion gate was not preregistered, so
the decision must be described as a calibration-informed operator decision rather than a passed
preregistered threshold. Do not change prompts, steps, guidance, seed, resolution, or method set
after opening test.
