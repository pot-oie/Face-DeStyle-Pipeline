# Multistyle pair-bank sequential Stage 2 review

All reviewed jobs completed technically: 8/8 3D, 12/12 Clay, and 10/10 Origami. Every Stage 2
record consumed its corresponding Stage 1 output and used seed 42, 28 steps, guidance 2.5, RGB,
and 1024-square output.

The visual decision is deliberately conservative:

| Style | Stage 1 | Stage 2 | Teacher | Available targets | Pending teacher | Identity-drift reject |
|---|---:|---:|---:|---:|---:|---:|
| 3D cartoon | 0 | 0 | 1 | 1 | 26 | 0 |
| Clay | 0 | 2 | 1 | 3 | 14 | 2 |
| Origami | 14 | 5 | 1 | 20 | 4 | 0 |

For 3D, a second FLUX edit mostly repeated or intensified animation geometry, large eyes, glossy
skin, and CGI lighting. It did not produce a reliable training target in the eight-case probe.

For Clay, Stage 2 sometimes removed material successfully, but this exposed the known failure
mode: a plausible human face can be invented rather than reconstructed. `matv2-clay-012` and
`matv2-clay-021` were retained; `013` and `015` were explicitly rejected for identity drift.

For Origami, Stage 1 already gave 14 usable face-domain reconstructions. Sequential Stage 2 added
five targets (`006`, `014`, `016`, `020`, and `024`), bringing the current pool to 19. This is close
to, but still below, the intended 20--40 useful-pair range. No LoRA should start merely to round 19
up to 20.

The next data action is a small closed-teacher pilot, not more Base FLUX passes:

- 3D: prioritize a diverse 5--8 source pilot because open sequential editing showed no rescue;
- Clay: prioritize 5--8 cases still marked `pending_closed_teacher`, and reject teacher results
  that invent a different person;
- Origami: try the five remaining hard cases (`008`, `021`, `025`, `027`, `029`) and retain only
  genuine improvements.

The filled local selection sheets and four-column review pages are stored under
`/Users/pot/Documents/大创/实验归档/multistyle-pair-bank-stage1-review-20260824`.

## Closed-teacher pilot addendum

A 2026-08-24 three-case pilot used OpenAI built-in reference-image generation through Codex as a
private-research teacher. The exact prompts are recorded in each local `closed-teacher/NOTES.md`.
All three pilot outputs were retained: `synthetic-3d-cartoon-006`, `matv2-clay-001`, and
`matv2-origami-008`. This changes the available-target counts to 1 for 3D, 3 for Clay, and 20 for
Origami. It is a promising teacher signal, not evidence that every teacher output will preserve
identity.

The 20-pair Origami ImageFolder dataset was built locally at
`/Users/pot/Documents/大创/实验归档/origami-lora-pairs-v1-20`. Its preview confirms 20 condition-target
pairs and its metadata records the selected target source for each pair. AutoDL preparation and
training-environment audit instructions are in `docs/AUTODL_ORIGAMI_LORA_PREP.md`.
