# Prompt for the next main implementation window

Continue `Face-DeStyle-Pipeline` from the current multistyle pair-bank and targeted-LoRA handoff.
Do not restart formal-v1 work and do not immediately resume LoRA training.

Local repository:
`/Users/pot/Github/Face-DeStyle-Pipeline`

Local dataset:
`/Users/pot/Documents/大创/Face-DeStyle-Data`

AutoDL project root:
`/root/autodl-tmp/face-destyle`

Latest repository commit should include at least:

- `1c24863 Add 3D destylization LoRA smoke path`
- `7c5901c Use restrained V2 prompt for full 3D pair run`
- `6438679 Align 3D LoRA training and evaluation prompts`

First read completely:

1. `AGENTS.md`
2. `docs/HANDOFF_MULTISTYLE_PAIR_BANK_AND_LORA.md`
3. `docs/research_context.md`
4. `docs/HANDOFF_MATERIAL_STYLE_EXTENSION_V1.md` as historical background only
5. `scripts/build_3d_lora_smoke_pairs.py`
6. `scripts/run_flux_kontext_probe.py`
7. `src/face_destyle/pipelines/flux_kontext_backend.py`
8. `configs/styles.yaml`
9. `configs/styles_3d_lora.yaml`

Important operator policy:

- prioritize a useful research narrative and working experiment over formal ceremony;
- do not require hashes, blind review, freeze markers, repeat scoring, complex archive validation,
  or formal statistics;
- preserve original images and outputs, use separate directories, and report sample counts honestly;
- local macOS cannot run CUDA or FLUX; heavy inference happens on AutoDL;
- preserve all existing user changes and inspect the worktree before editing.

Research context:

The senior student's paper is *Learning to Stylize by Learning to Destylize*
(`/Users/pot/Documents/大创/2509.05970v2.pdf`). This is a small face-domain exploratory subproject,
not the paper's official implementation. The useful inherited ideas are reverse data construction,
progressive candidate generation/filtering, and LoRA only after high-quality pairs exist.

The project found that FLUX Kontext is much stronger than the SDXL prompt/Canny baselines on comic,
ink, and watercolor, but still struggles with geometry/material-entangled 3D. The material extension
also found clay difficult, while needle felt was often handled by one-pass prompting.

Completed negative LoRA result:

- 24 synthetic natural-target/3D-condition pairs were generated;
- eight conservative pairs (`004,005,009,010,016,019,021,023`) trained a rank-16, 200-step LoRA;
- it loaded successfully and visibly changed five held-out 3D pilot outputs;
- it did not improve destylization and instead stayed closer to the original 3D images;
- comparison sheets are in
  `/Users/pot/Documents/大创/实验归档/3d-lora-base-vs-adapted-pilot5-comparison`;
- preserve this as a negative baseline and do not continue its checkpoint.

A proposed 19-pair retraining did not start because a later shell could not find `accelerate`.
Do not fix that environment or train the 19-pair set yet. The active strategy is to construct pairs
from genuine styled source images rather than relying on lucky natural-to-3D synthesis.

Current raw inventory is approximately 67 3D, 12 clay, 12 needle felt, 60 comic, 61 ink, and 92
watercolor files. A separate task is generating additional clay, needle-felt, and origami sources.

Your first concrete task:

1. inspect the 67 local 3D raw files without altering them;
2. create a compact contact sheet and select roughly 40--50 diverse, usable 3D portrait sources;
3. reserve 5--10 sources for later Base-vs-LoRA qualitative testing;
4. keep a simple selected-file list; no cryptographic or formal validation;
5. design or implement the smallest useful path for candidate generation:
   - Base FLUX Stage 1 from the styled source;
   - true sequential Stage 2 using the Stage 1 output as input;
   - a separate location for closed-source teacher outputs;
6. build review sheets with
   `styled source | FLUX Stage 1 | sequential Stage 2 | closed teacher`;
7. do not train another LoRA until the operator has selected roughly 20--40 reliable pairs.

The old material-extension Stage 2 edited the original source again. Do not call that a true second
generation. The new sequential path must explicitly consume the Stage 1 output.

The intended later training direction is one style-specific LoRA each for 3D, clay, and origami.
Needle felt remains a prompt-only control unless new evidence shows persistent failure. Closed-source
models may be used as private teacher target generators; the final LoRA remains a local FLUX adapter.

Run Ruff, complete pytest, relevant script `--help`, and `git diff --check` for code changes. Push
only when the operator's current authorization allows it. Do not start model inference from local
macOS and do not launch unrelated LoRA, Multi-ControlNet, pose, depth, Qwen full evaluation, or v2
formal experiments.

