# Method

The proposed pipeline performs face-domain destylization in two prompt-guided refinement stages.
Both stages request a natural photographic result while preserving identity, pose, composition, and
background structure. The experiment varies generic versus style-category prompts and structural
conditioning: none, global Canny, face/background region-aware Canny, and optional pose.

The current repository implements only orchestration and lightweight controls. `copy` verifies data
flow without changing pixels. Diffusion inference, production face segmentation, and pose control
remain explicit AutoDL tasks and must record model revision, sampler, seed, resolution, and prompts.

Accepted candidates pass content and style-removal thresholds plus an optional identity threshold.
Triplets use the accepted destylized output as content, a different source in the same category as
style reference, and the original stylized input as target.
