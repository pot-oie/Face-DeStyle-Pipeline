# Origami hard-pair V2 independent review (2026-08-25)

## Outcome

The returned expansion is large and clean enough for the next training experiment. The generator
accepted 30 pairs; independent full-frame review accepts 28 of them for training. Together with the
original 23 strict pairs, the next dataset should contain 51 pairs.

Do not generate another expansion before testing this 51-pair intervention.

## Inspected artifact

- working tree:
  `/Users/pot/Documents/大创/Face-DeStyle-Data/extensions/origami_hard_pairs_v2`
- returned ZIP:
  `/Users/pot/Desktop/origami-hard-pairs-v2-generation-and-review.zip`
- 36 RGB 1254x1254 source PNGs
- 30 RGB 1254x1254 generator-accepted teacher PNGs
- 34 initial teacher attempts and 6 targeted retries
- 6 source or teacher candidates rejected by the generation window
- 13 review contact sheets
- no protected Origami holdout ID in the package metadata

The files decode correctly and the source/target IDs are internally consistent. The ZIP contains no
macOS resource-fork entries. The package is suitable as private experimental input.

The delivered CSV column names differ slightly from the requested template, and their paths are
absolute rather than relative. This is not an image-quality blocker. Normalize the schema and paths
when creating the repository-owned V2 selection manifest; do not train directly from the delivered
CSV.

## Independent selection

Accept the generator's selected set except:

- `origami-hard-v2-021`: the source is explicitly a bust on a circular pedestal, while the teacher
  removes the pedestal and turns the subject into an unsupported half-body portrait;
- `origami-hard-v2-023`: the source has a visible sculptural base, while the teacher changes it into
  a standing/cropped human torso without the base.

Both remove paper well, but they violate the expansion prompt's requirement to preserve crop and
composition and to retain a physically plausible support where one exists. This is the same class
of drift that caused candidate `017` to be rejected upstream.

The 28 training-accepted IDs are:

```text
origami-hard-v2-001, origami-hard-v2-002, origami-hard-v2-004,
origami-hard-v2-005, origami-hard-v2-006, origami-hard-v2-007,
origami-hard-v2-008, origami-hard-v2-009, origami-hard-v2-010,
origami-hard-v2-011, origami-hard-v2-012, origami-hard-v2-013,
origami-hard-v2-015, origami-hard-v2-016, origami-hard-v2-019,
origami-hard-v2-020, origami-hard-v2-022, origami-hard-v2-024,
origami-hard-v2-025, origami-hard-v2-026, origami-hard-v2-027,
origami-hard-v2-028, origami-hard-v2-029, origami-hard-v2-030,
origami-hard-v2-031, origami-hard-v2-032, origami-hard-v2-033,
origami-hard-v2-036
```

Across these 28, overlapping tags include 11 elderly-wrinkle, 9 profile, 9 hair/headwear, 8 strong
shadow, 7 dark-skin, 7 scalp/neck, 7 clothing/bust, 7 bald, 6 beard, 6 compound, and 5 geometric
exaggeration examples. The set covers the observed checkpoint-100 failure regions and is not merely
an easy-count expansion.

## Next experiment

Build a new 51-pair ImageFolder dataset from the original 23 pairs plus these 28. Keep the original
six holdouts out of training. Use a common strong full-subject instruction plus tag-specific region
clauses, then train a fresh rank-16 adapter from the base Kontext model. With effective batch 4,
checkpoints at 50/100/150/200 correspond to about 3.9/7.8/11.8/15.7 effective epochs; stop at 200
for the first V2 comparison. Compare those checkpoints on the unchanged six-image holdout set before
considering more data or optimization changes.
