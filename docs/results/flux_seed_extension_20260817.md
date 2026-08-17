# FLUX seed-stability extension audit — 2026-08-17

## Scope

This is a stochastic-stability extension of the frozen 20-source native-1024
`FLUX.1-Kontext-dev` pilot. It is not part of the primary seed-42 method comparison, a parameter
sweep, or evidence of method quality. No evaluation score or acceptance decision is reported here.

## Returned archives

- initial archive: `seed-43-and-partly-44.zip`, containing seed 43 complete and the first five
  seed-44 records; SHA-256
  `39e5dd14655fd0d8c6893003c2b7bbbc5a7c07afa0947925377ffbbddae47465`;
- completed seed-44 archive: `flux-kontext-native1024-seed-44-complete.zip`; SHA-256
  `9c2042df4b4ee81074467b0cc2f395d044421f719e49a5cb3d84140d5fcce591`;
- both ZIP CRC checks passed;
- every returned image decoded as an RGB PNG at 1024x1024;
- recorded generation failures: zero.

## Run completeness

| Seed | Valid records | Style coverage | Status |
|---:|---:|---|---|
| 43 | 20/20 | five each for `3d_cartoon`, `comic`, `ink`, `watercolor` | complete |
| 44 | 20/20 | five each for `3d_cartoon`, `comic`, `ink`, `watercolor` | complete after resume |

Seed 43 used the declared `flux1_kontext_dev_prompt_edit_bf16_offloaded` backend and averaged
110.36 seconds of recorded inference per image. Seed 44 was resumed from its first five valid
records and completed the remaining fifteen without a recorded failure. The complete archive
contains 20 unique source records and 20 valid 1024x1024 RGB PNGs. It passed ZIP CRC validation and
has SHA-256 `9c2042df4b4ee81074467b0cc2f395d044421f719e49a5cb3d84140d5fcce591`.

## Use boundary

Preserve all returned outputs and stop adding seeds. Analyze seed sensitivity only after the
predeclared comparison is complete; do not select a favorable seed or tune settings from these
images.
