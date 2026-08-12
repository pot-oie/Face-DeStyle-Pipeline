# Experiment plan

1. Curate and document a small authorized multi-style face set; freeze train/calibration/test splits.
2. Implement and verify one pinned diffusers baseline on AutoDL.
3. Compare generic and category-adaptive prompts with identical seeds.
4. Add global Canny, then face/background region-aware Canny, then optional pose.
5. Implement DINO/CLIP, ArcFace, and VLM evaluators and validate them against human labels.
6. Calibrate thresholds, freeze settings, run the test split, and manually audit accepted samples.
7. Build deterministic same-style, different-source triplets and publish only permissible metadata.

Start with a small factorial pilot. Record seed, source ID, prompt version, model revision, scheduler,
steps, guidance, control strength, resolution, runtime, and failures for every generation. Do not
claim improvements from the copy backend or smoke metric.
