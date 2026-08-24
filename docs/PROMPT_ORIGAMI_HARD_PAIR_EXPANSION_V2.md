# Prompt for the Origami hard-pair generation and curation window

Continue the independent image-generation and material-curation task for
`Face-DeStyle-Pipeline`. This window owns only the next Origami data-pair expansion. Generate and
organize the images locally; do not run CUDA, AutoDL training, LoRA inference, or formal-v1 work.

Use the available `imagegen` skill and read its instructions before generating or editing images.
Generate directly without asking the operator to approve each image. Inspect every output
visually, retry an unacceptable teacher target at most once, and keep moving until the requested
bank is complete or a genuine tool limit blocks progress.

## Context and objective

The first strict Origami dataset contained 23 source/teacher pairs. A rank-16 FLUX Kontext LoRA
trained on it peaked at checkpoint 100. On six untouched holdouts, Base FLUX strictly passed about
1/6 and checkpoint 100 passed 3/6. Checkpoints 200 and 300 did not rescue more cases and introduced
more facial drift. The remaining failures were concentrated in:

- large folded-paper hair, hair ornaments, hoods, and headscarves;
- elderly skin combined with folded beards, moustaches, or gray hair;
- bald scalps, ears, necks, and faces defined by severe paper planes;
- folded-paper clothing, shoulders, busts, and pedestals that survive after the face naturalizes;
- compound hard cases with dark skin, strong shadows, three-quarter views, or unusual gaze.

The goal is not simply “more Origami images.” Create a new bank of difficult, fully paired examples
that teaches removal of paper from the entire subject while preserving identity and composition.

## Protected evaluation data

The six existing holdouts are `matv2-origami-002`, `007`, `011`, `018`, `023`, and `030`. They must
remain evaluation-only. Do not copy them, edit them into targets, rename them as training data, or
include them in the new pair bank. They may be viewed only to understand failure categories. Every
new source must depict a different fictional identity and a materially different composition.

## Output root and layout

Create everything under:

`/Users/pot/Documents/大创/Face-DeStyle-Data/extensions/origami_hard_pairs_v2/`

Use this layout:

```text
raw/sources/
teacher/accepted/
teacher/rejected/
review/contact-sheets/
manifests/source_candidates.csv
manifests/selected_pairs.csv
provenance/NOTES.md
```

Use IDs `origami-hard-v2-001` through `origami-hard-v2-036`. Do not overwrite the existing
`material_styles_v2` bank.

## Stage A: generate 36 difficult styled sources

Generate 36 square 1024x1024 fictional portrait sources. Every source must show one primary person
and unmistakable three-dimensional origami construction across the subject, not a photographic face
with a paper texture overlay. Distribute the bank approximately as follows; categories may overlap:

- 8 with large paper hair, elaborate buns, layered curls, hoods, or headscarves;
- 7 elderly subjects with paper wrinkles plus beard, moustache, or gray hair structure;
- 8 dominated by paper clothing, layered shoulders, full bust, or a visible pedestal;
- 5 bald or closely shaven subjects with severe scalp, ear, face, and neck planes;
- 8 compound hard cases combining strong geometry with dark skin, dramatic lighting, side pose,
  unusual gaze, accessories, or complex garments.

Across the 36, vary apparent age, gender presentation, skin/paper palette, face shape, body type,
pose, gaze, expression, crop, hairstyle, clothing, accessories, lighting, and background. Include
front, three-quarter, profile-leaning, upward, and downward poses. Avoid repeated faces and avoid
changing only colors between images.

Source requirements:

- visibly folded paper across face, hair/headwear, neck, and clothing or bust;
- coherent portrait anatomy with two readable eyes when the pose permits;
- no real person, celebrity, protected character, logo, watermark, caption, UI, collage, or text;
- no broken hands or extra people; hands are preferably outside the crop;
- no flat paper-texture overlay, ordinary illustration, plastic CGI, clay, or stone;
- retain the original full-resolution PNG.

Record the exact generation prompt and available model metadata for each source in
`provenance/NOTES.md`. Build a source contact sheet and reject obvious anatomical failures or near
duplicates before target generation.

## Stage B: create identity-preserving natural teacher targets

For each accepted source, use it as the reference image for a fresh identity-preserving edit. The
target must be a natural photorealistic camera portrait of the same fictional person. Use a prompt
adapted to the visible subject, based on this policy:

> Replace every folded-paper surface across the skin, scalp, hair, headwear, eyebrows, eyelashes,
> ears, beard, moustache, neck, clothing, shoulders, bust, and pedestal with biologically plausible
> human features, individual hair strands, real accessories, naturally draped woven fabric, and a
> physically plausible support where present. Preserve the same fictional identity, apparent age,
> skin tone, body type, facial proportions, pose, gaze, expression, silhouette, hairstyle or
> baldness, accessory shapes, garment layout, crop, background, palette, and lighting. Do not
> rejuvenate, beautify, change gender presentation, open or close the mouth, redirect the eyes, or
> redesign the composition. No paper, folded facets, creases, cut edges, polygonal skin, sculpture,
> mannequin, CGI, text, or watermark.

Customize the list of regions and identity details to each image. Generate one teacher target first.
If it retains paper or causes identity drift, make at most one targeted retry. Store accepted targets
under `teacher/accepted/` with the same basename as the source. Store failed attempts under
`teacher/rejected/`; never silently promote them.

## Strict full-frame acceptance rule

Select 24–30 pairs from the 36 sources. A target passes only when all of the following hold:

1. face and skin are natural, without polygonal or embossed marks;
2. hair, headwear, beard, moustache, ears, scalp, and neck contain no paper construction;
3. clothing, shoulders, bust, and pedestal contain no residual folded-paper material;
4. identity evidence, age, skin tone, pose, gaze, expression, face shape, and body type remain
   recognizably consistent;
5. crop, garment layout, major accessories, background, palette, and lighting remain stable;
6. there is no beautification, rejuvenation, anatomy error, text, watermark, or new person.

Inspect the complete frame, not just the face. Angular folds in real woven fabric are allowed only
when they read clearly as fabric rather than paper. If uncertain, reject the pair.

## Manifests and review artifacts

Create `source_candidates.csv` with:

```text
source_id,source_path,difficulty_tags,source_decision,notes
```

Create `selected_pairs.csv` with:

```text
source_id,condition_path,target_path,decision,difficulty_tags,reviewer_notes
```

Use `accept`, `reject_source`, `reject_teacher_residual`, or `reject_identity_drift` as decisions.
Paths must be relative to the output root. Create readable contact sheets with at least:

`styled source | accepted teacher target`

and label every row with its source ID. Also create category-specific sheets for hair/headwear,
elderly/beard, scalp/neck, clothing/bust, and compound cases when enough rows exist.

## Completion handoff

Do not modify the repository training manifest and do not start LoRA training. When complete, report:

- number of generated sources;
- number of accepted pairs and rejection counts by reason;
- distribution of accepted difficulty tags;
- exact output root;
- any IDs that remain ambiguous;
- paths to `selected_pairs.csv`, contact sheets, and `provenance/NOTES.md`.

Package the complete `origami_hard_pairs_v2` directory as:

`/Users/pot/Desktop/origami-hard-pairs-v2-generation-and-review.zip`

The next main window will independently inspect the images, merge only accepted pairs with the
original 23-pair dataset, strengthen per-example instructions, build an approximately 47–53-pair
ImageFolder dataset, and prepare a fresh rank-16 training run. Do not claim completion unless the
images, manifests, notes, contact sheets, and ZIP all exist.
