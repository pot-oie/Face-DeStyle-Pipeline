# FLUX seed-stability extension audit — 2026-08-17

## Scope

This is a stochastic-stability extension of the frozen 20-source native-1024
`FLUX.1-Kontext-dev` pilot. It is not part of the primary seed-42 method comparison, a parameter
sweep, or evidence of method quality. No evaluation score or acceptance decision is reported here.

## Returned archive

- archive: `seed-43-and-partly-44.zip` (stored outside Git with the experiment archives);
- SHA-256: `39e5dd14655fd0d8c6893003c2b7bbbc5a7c07afa0947925377ffbbddae47465`;
- ZIP CRC: passed;
- decoded outputs: 25 RGB PNG files at 1024x1024;
- recorded generation failures: zero.

## Run completeness

| Seed | Valid records | Style coverage | Status |
|---:|---:|---|---|
| 43 | 20/20 | five each for `3d_cartoon`, `comic`, `ink`, `watercolor` | complete |
| 44 | 5/20 | five `3d_cartoon` only | interrupted after the fifth record |

Seed 43 used the declared `flux1_kontext_dev_prompt_edit_bf16_offloaded` backend and averaged
110.36 seconds of recorded inference per image. Seed 44 averaged 109.64 seconds over its five
returned records. The seed-43 log ends with a successful 20-record summary; the seed-44 log stops
after the fifth image without a final run summary. The five returned seed-44 records themselves are
valid, so this is treated as an interrupted resumable run rather than five successes plus fifteen
model failures.

## Use boundary

Preserve all returned outputs. If seed 44 is resumed, use the identical manifest, model, prompts,
steps, guidance, resolution, and seed with `--resume`, and package the completed directory without
overwriting this archive. Analyze seed sensitivity only after the predeclared comparison is
complete; do not select a favorable seed or tune settings from these images.
