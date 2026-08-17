# Formal generation server status — 2026-08-17

Two read-only workspace inventories at clean commit `b5a2f36` report completion of the planned
formal generation batch. Both machines validated all 120 frozen source files and SHA-256 values.

| Method | Split coverage | Records | Output paths present | Failures | Seed |
|---|---|---:|---:|---:|---:|
| SDXL generic prompt-only | calibration 40 + test 60 | 100 | 100 | 0 | 42 |
| SDXL adaptive prompt-only | calibration 40 + test 60 | 100 | 100 | 0 | 42 |
| SDXL Global Canny | calibration 40 + test 60 | 100 | 100 | 0 | 42 |
| SDXL Region Canny | calibration 40 + test 60 | 100 | 100 | 0 | 42 |
| FLUX Kontext native 1024 | calibration 40 | 40 | 40 | 0 | 42 |

The four SDXL archives exist on host `autodl-container-3l8zswwuan-23a8497b`; the FLUX calibration
archive exists on host `autodl-container-2v5jqfbf4z-a7217362`. These are server-inventory facts,
not yet local archive validation. Download each ZIP together with its `.zip.sha256`, verify the
sidecar and ZIP CRC locally, and preserve the server run directories until that succeeds.

No further generation is currently justified. Keep the 60-source SDXL test subset uninspected and
unevaluated. The next scientific step is calibration-only DINO/CLIP/ArcFace evaluation plus two
rounds of method-hidden human scoring across the four SDXL methods and FLUX. Qwen2.5-VL-3B is not
repeated broadly because the pilot established method-insensitive score compression; larger-model
audits remain limited to human/metric disagreements.
