# Formal private-research source manifest v1

This manifest freezes the 120-source study inventory on 2026-08-17: 20 pilot/debug sources,
40 calibration sources, and 60 held-out test sources. Each split contains the same four style
categories (`3d_cartoon`, `comic`, `ink`, and `watercolor`) with 5/10/15 sources per category.

The raw images remain outside Git below a separately managed `Face-DeStyle-Data` root. This folder
is a runnable integrity and split declaration, not an image redistribution package. In particular,
synthetic sources marked `not_publicly_licensed` remain private-research-only pending a separate
project redistribution review. The public provenance file is deliberately minimal and does not
contain local absolute paths, generation prompts, image download URLs, embeddings, or face
templates.

Validation at freeze time established:

- all 120 declared files existed and matched their SHA-256 values;
- `source_id`, `source_group_id`, and file SHA-256 had zero overlap between pilot, calibration,
  and test;
- no pair from the batch-2 candidate near-duplicate report was jointly selected into test;
- calibration and test curation reports contained no validation errors.

The test split is sealed. Do not generate from it, inspect method outputs from it, or change
thresholds from it until blinded human calibration has been completed and the full method/routing
configuration has been frozen. Corrections require a documented amendment or a new manifest
version; never silently rewrite this directory after test inference.

Resolve and checksum-validate `asset_path` against the common external data root without running a
model:

```bash
export FACE_DESTYLE_DATA_ROOT=/path/to/Face-DeStyle-Data
python -c 'from face_destyle.data.manifests import load_dataset_manifest; print(len(load_dataset_manifest("data/manifests/formal-v1/inputs.jsonl", data_root=None)))'
```

`checksums.sha256` uses the same common data-root-relative paths. `provenance.public.jsonl` records
only fields safe to keep with the code; the richer private provenance remains with the external
data archive.
