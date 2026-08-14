# Method

The proposed pipeline performs face-domain destylization in two prompt-guided refinement stages.
Both stages request a natural photographic result while preserving identity, pose, composition, and
background structure. The completed pilot varies generic versus style-category prompts and
structural conditioning: none, global Canny, and face/background region-aware Canny. Pose remains a
declared later extension rather than the active comparison.

The current repository implements orchestration, lightweight controls, a prompt-only SDXL img2img
baseline, global Canny ControlNet, and a region-aware Canny backend with a completed one-source GPU
runtime and condition-map smoke test plus a completed 20-source pilot comparison.
`copy` verifies data flow without changing pixels. Generation uses the configured model revisions,
input image, positive/negative prompt, strength, steps, guidance, size, and seed. Global Canny saves
the exact condition image. Region Canny uses registered SegFormer face parsing, retains full-strength
edges over parsed head labels, attenuates background edges, and saves the global edges, parsed mask,
and final composite condition. The pilot did not show a stable Region Canny improvement over global
Canny, so that negative/mixed outcome must be analyzed before any new structural extension. Every
run records model revision, seed, resolution, prompts, conditions, and core sampling settings in its
`DestylizationRecord`.

The next exploratory method is `flux1_kontext_dev_prompt_edit_bf16_offloaded`: original BF16
FLUX.1 Kontext native image editing with CPU/model offload, batch size 1, and no Canny, Depth, Pose,
LoRA, or quantization. It is a generator-capability probe against the existing SDXL Base adaptive
prompt-only output, not a claim that seeds define matched noise across architectures. The first run
is fixed to one preselected source from each primary style at 768×768. Only successful engineering
validation and a non-cherry-picked capability signal may justify a fixed 20-source extension.

Accepted candidates pass content and style-removal thresholds plus an optional identity threshold.
Triplets use the accepted destylized output as content, a different source in the same category as
style reference, and the original stylized input as target.
