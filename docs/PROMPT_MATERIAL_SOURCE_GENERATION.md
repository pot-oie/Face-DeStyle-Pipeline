# Prompt for the existing material-source generation window

Continue the independent source-material generation task for `Face-DeStyle-Pipeline`. This window
should generate styled input artworks only. Do not perform destylization, natural reconstruction,
LoRA training, scoring, or formal evaluation.

Shared dataset root:
`/Users/pot/Documents/大创/Face-DeStyle-Data`

The main research has enough 3D candidates already (about 67 raw files). Prioritize expanding the
material styles that are still small:

- generate 24 new fictional clay/terracotta portrait sources;
- generate 30 new fictional origami/paper-fold portrait sources;
- generate 12 new fictional needle-felt portrait sources.

Store them separately under:

- `extensions/material_styles_v2/raw/clay/`
- `extensions/material_styles_v2/raw/origami/`
- `extensions/material_styles_v2/raw/needle_felt/`

Use stable names such as:

- `matv2-clay-001.png` through `matv2-clay-024.png`;
- `matv2-origami-001.png` through `matv2-origami-030.png`;
- `matv2-needle-felt-001.png` through `matv2-needle-felt-012.png`.

Image requirements:

- 1024x1024 square when possible;
- one primary fictional person with a clearly visible face;
- unmistakable declared material style, not a texture overlay on a photograph;
- vary apparent age, gender presentation, skin/material color, face shape, eye size, pose, crop,
  expression, clothing, lighting, and background complexity;
- include both restrained and strongly stylized geometry;
- no real named person, public figure, protected character, studio imitation, logo, watermark,
  caption, UI, collage, or text;
- avoid near duplicates and do not reuse the same face/composition with only a color change.

Style-specific cues:

- clay: modeled volume, hand-shaped or molded surfaces, dry clay/terracotta material, occasional
  tool marks; avoid glossy plastic or ordinary stone sculpture;
- origami: visibly folded paper planes and facets forming the face, paper edges and creases, coherent
  three-dimensional portrait structure; avoid a flat paper-texture overlay;
- needle felt: visible wool fibers, needle-punched handmade construction, soft sculptural volume;
  avoid turning only the clothing or background into fabric.

Create one simple contact sheet per style and visually remove obvious generation failures. Keep a
small text or JSONL note containing filename, style, model, prompt, and seed when available. Do not
add hashes, blind review, elaborate provenance machinery, or formal acceptance statistics. These
are private project-generated source artworks for later multi-model reconstruction and pair
selection.

Do not generate natural target photographs in this window. The main pipeline will later send each
styled source through Base FLUX, true sequential Stage 2, and possibly a closed-source teacher, then
select the best natural reconstruction as the training target.

