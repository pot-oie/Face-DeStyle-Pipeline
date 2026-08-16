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

The completed exploratory method is `flux1_kontext_dev_prompt_edit_bf16_offloaded`: original BF16
FLUX.1 Kontext native image editing with CPU/model offload, batch size 1, native 1024x1024 output,
and no Canny, Depth, Pose, LoRA, or quantization. It completed all 20 frozen pilot sources in one
pipeline-loading session. It is a generator-capability probe against the existing SDXL Base adaptive
prompt-only outputs, not a claim that equal seeds define matched noise across architectures. Because
SDXL used 768x768 and FLUX used its native resolution, the comparison is intentionally exploratory.
Unblinded review found encouraging media conversion on comic, ink, and watercolor sources and a
consistent failure to remove rendered 3D geometry/materials. These observations must be checked by
the frozen human rubric and formal metrics before they are used as results.

Accepted candidates pass content and style-removal thresholds plus an optional identity threshold.
Triplets use the accepted destylized output as content, a different source in the same category as
style reference, and the original stylized input as target.
