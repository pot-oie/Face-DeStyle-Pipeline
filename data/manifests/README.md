# Frozen dataset manifests

This directory is reserved for versioned, sanitized declarations such as `formal-v1/`. Do not copy
raw images, private provenance, absolute local paths, embeddings, credentials, or unreviewed notes
into the code repository.

A runnable `inputs.jsonl` contains one accepted source per line:

```json
{"id":"example","source_id":"example","source_group_id":"example-group","asset_path":"raw/comic/example.png","style_category":"comic","split":"pilot","sha256":"64-lowercase-hex-characters","qc_status":"accepted"}
```

`asset_path` is resolved against a separate data directory supplied with `--data-root` or the
`FACE_DESTYLE_DATA_ROOT` environment variable. The loader verifies path containment, file presence,
SHA-256, unique source IDs, and source-group split isolation before inference.

A frozen version should include:

- `README.md` with scope, limitations, freeze date, and counts;
- `dataset.json` with a schema/version identifier and generating code commit;
- `inputs.jsonl` with the minimal runnable declarations;
- `provenance.public.jsonl` with only approved public provenance fields;
- `splits.json` with frozen source-group assignments;
- `checksums.sha256` using paths relative to the external data root.

Create `formal-v1/` only after the dataset owner confirms the QC decisions, redistribution review,
and pilot/calibration/test freeze. Later corrections should be a new explicit manifest version or a
documented amendment, not a silent rewrite after test results are known.

`pilot_runtime_manifest.jsonl` is a small checksum-pinned exception for the already frozen 20-source
pilot used by the completed SDXL and FLUX runs. Its raw images remain in the separate
`Face-DeStyle-Data` root. The split is pilot/debug and must not be reassigned to calibration or
held-out test.
