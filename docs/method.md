# Method

The proposed pipeline performs face-domain destylization in two prompt-guided refinement stages.
Both stages request a natural photographic result while preserving identity, pose, composition, and
background structure. The experiment varies generic versus style-category prompts and structural
conditioning: none, global Canny, face/background region-aware Canny, and optional pose.

The current repository implements orchestration, lightweight controls, and a first prompt-only
SDXL img2img baseline. `copy` verifies data flow without changing pixels. The SDXL baseline uses the
configured model revision, input image, positive/negative prompt, strength, steps, guidance, size,
and seed; it does not yet use Canny, face regions, or pose. Production face segmentation and pose
control remain explicit AutoDL tasks. Every run records model revision, seed, resolution, prompts,
and core sampling settings in its `DestylizationRecord`.

Accepted candidates pass content and style-removal thresholds plus an optional identity threshold.
Triplets use the accepted destylized output as content, a different source in the same category as
style reference, and the original stylized input as target.
