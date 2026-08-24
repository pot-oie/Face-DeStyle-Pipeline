# Multistyle pair-bank sequential Stage 2 review

All reviewed jobs completed technically: 8/8 3D, 12/12 Clay, and 10/10 Origami. Every Stage 2
record consumed its corresponding Stage 1 output and used seed 42, 28 steps, guidance 2.5, RGB,
and 1024-square output.

The visual decision is deliberately conservative:

| Style | Stage 1 | Stage 2 | Teacher | Strict targets | Pending teacher | Rejected open targets |
|---|---:|---:|---:|---:|---:|---:|
| 3D cartoon | 0 | 0 | 1 | 1 | 26 | 0 |
| Clay | 0 | 0 | 1 | 1 | 14 | 4 |
| Origami | 0 | 0 | 5 | 5 | 0 | 19 |

For 3D, a second FLUX edit mostly repeated or intensified animation geometry, large eyes, glossy
skin, and CGI lighting. It did not produce a reliable training target in the eight-case probe.

For Clay, Stage 2 sometimes removed facial material successfully, but this exposed two failure
modes: clay clothing remained on `012` and `021`, while `013` and `015` invented a substantially
different person. None is a strict full-portrait natural target.

For Origami, Stage 1 and Stage 2 often naturalized the central face while leaving folded-paper hair,
headwear, clothing, or bust geometry. An initial face-only review mistakenly counted 19 of these as
usable. The corrected full-portrait review rejects all 19 rather than teaching a LoRA to preserve
the residual material.

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
`matv2-origami-008`. Four additional Origami hard cases (`021`, `025`, `027`, and `029`) were then
generated with stricter whole-portrait prompts and accepted after source/Stage 1/Stage 2/teacher
comparison. Origami therefore has five strict targets, while 3D and Clay each have one. This is a
promising teacher signal, not evidence that every teacher output will preserve identity.

The rejected 20-pair Origami ImageFolder preview remains locally at
`/Users/pot/Documents/大创/实验归档/origami-lora-pairs-v1-20` with a `DO_NOT_TRAIN.md` warning. It is
preserved as evidence of the review mistake and must not be uploaded or trained.
