# Showcase refinement 20-source review

## Outcome

The post-closure showcase extension completed 20/20 Stage 1 outputs and 20/20 true sequential
Stage 2 outputs with no recorded runner failure. It used original-BF16 FLUX.1-Kontext-dev, seed 42,
28 steps, guidance 2.5, no LoRA, and style-specific prompts for 10 selected 3D-cartoon and 10
selected Needle-felt sources.

This run was designed to obtain a small reserve of presentation candidates after the 137-source
study had already established the conservative processing routes. It is not added to the 137-source
pass counts and is not a formal evaluation. The exact visual decisions are in
[`showcase_refinement_20_review_20260827.csv`](showcase_refinement_20_review_20260827.csv).

## Selected display candidates

| Style | Primary selections | Backups | Interpretation |
|---|---|---|---|
| 3D cartoon | `synthetic-3d-cartoon-003` Stage 2 | `017`, `021`, and `024` Stage 1 | Targeted anatomy language improves some sources, but most of the ten still read as polished animation or CG. Use this as a hard-style refinement demonstration, not a universal 3D success claim. |
| Needle-felt | `matv2-needle-felt-003`, `005`, `synthetic-needle-felt-011`, `012` Stage 1 | none needed | Portrait-oriented semantic reconstruction can replace bust/support construction with coherent shoulders and clothing. Stage 2 usually adds sharpness rather than a useful rescue. |

For Needle-felt, the display contract differs from the strict full-frame rule used in the 137-source
review: a handmade bust may be reconstructed as a plausible waist-up human portrait. The selected
examples preserve the depicted age, pose, gaze, hairstyle, palette, and broad facial evidence, but
they are model reconstructions rather than recovered true appearances.

## Practical route learned from the extension

- Use the finalized Base Stage 1 route for Comic, Ink, and Watercolor.
- Use true sequential Stage 2 only for reviewed Clay rescues.
- For 3D cartoon, try anatomy-aware reconstruction only as a reviewed showcase or fallback branch;
  the closed-teacher `synthetic-3d-cartoon-006` remains the clearest ceiling example.
- For Needle-felt busts, use portrait-oriented semantic reconstruction and prefer Stage 1 when it
  already creates coherent skin, hair, shoulders, and fabric.
- Keep Origami V1 checkpoint 100 as a separate limited adapter at about 3/6 on its six fixed
  qualitative holdouts.

## Local visual package

The derived comparison panels are outside Git at:

```text
/Users/pot/Documents/大创/实验归档/portfolio-showcase-20260827
```

They contain only presentation composites and a short local guide. Original inputs, model weights,
bulk outputs, and returned run folders remain in their existing local locations.
