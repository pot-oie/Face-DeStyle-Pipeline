# Data policy

`raw/` and `processed/` are intentionally ignored except for `.gitkeep`. Do not commit complete
datasets, private metadata, or any face image without documented authorization from the rights
holder and, where applicable, the depicted person. Dataset licenses, consent, privacy, copyright,
retention, and deletion obligations remain the researcher's responsibility.

Only tiny synthetic or explicitly redistributable examples belong in `samples/`. Even then, record
provenance and terms. Large data should live on an AutoDL data disk and have a separate backup.

Frozen, sanitized declarations may be committed under `manifests/<version>/`; raw assets remain in
a separate data root. Manifest `asset_path` values are relative to `FACE_DESTYLE_DATA_ROOT`, never
developer-machine or AutoDL absolute paths. See [`manifests/README.md`](manifests/README.md).
