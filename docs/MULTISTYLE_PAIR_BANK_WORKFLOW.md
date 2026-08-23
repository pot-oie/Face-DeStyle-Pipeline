# Multistyle reconstruction pair-bank workflow

This is the lightweight operational path for 3D, clay, and origami reconstruction candidates.
The source banks are still being expanded, so do not freeze a final count or start LoRA training
yet.

## 1. Keep one readable source list per style

Use CSV columns:

```csv
source_id,asset_path,style_category,role,notes
example-001,raw/3d_cartoon/example-001.png,3d_cartoon,candidate,clear frontal portrait
example-002,raw/3d_cartoon/example-002.png,3d_cartoon,holdout,hard geometry case
example-003,raw/3d_cartoon/example-003.png,3d_cartoon,rejected,tiny face
```

`asset_path` is relative to `--data-root`. `candidate` rows are used for reconstruction generation,
`holdout` rows are reserved for later Base-vs-LoRA comparison, and `rejected` rows remain visible in
the inventory contact sheet. This list intentionally does not require hashes or a formal freeze.

## 2. Prepare the non-overwriting run layout

```bash
python scripts/build_pair_bank_review.py \
  --source-list /path/to/3d_source_bank.csv \
  --data-root "$FACE_DESTYLE_ROOT/data" \
  --run-dir "$FACE_DESTYLE_ROOT/outputs/pair-bank-3d-v1"
```

This creates separate `stage1/images`, `stage2-sequential/images`, and
`closed-teacher/images` directories. It also creates an inventory contact sheet, four-column review
pages, a closed-teacher notes file, and `review/target_selection.csv`. Rerunning refreshes derived
contact/review sheets but does not overwrite model outputs or an existing selection CSV.

## 3. Generate Base FLUX Stage 1

The pair-bank source list can be passed directly to the existing runner. Repeat
`--required-style` only when one list intentionally contains multiple styles.

```bash
python scripts/run_flux_kontext_probe.py \
  --source-list /path/to/3d_source_bank.csv \
  --data-root "$FACE_DESTYLE_ROOT/data" \
  --required-style 3d_cartoon \
  --prompt-stage stage1 \
  --probe-stage batch \
  --model-dir "$FACE_DESTYLE_ROOT/models/diffusion/FLUX.1-Kontext-dev" \
  --download-manifest /path/to/download-manifest.json \
  --hash-manifest /path/to/hash-manifest.json \
  --output-dir "$FACE_DESTYLE_ROOT/outputs/pair-bank-3d-v1/stage1/images" \
  --records-output "$FACE_DESTYLE_ROOT/outputs/pair-bank-3d-v1/stage1/records.jsonl" \
  --failures-output "$FACE_DESTYLE_ROOT/outputs/pair-bank-3d-v1/stage1/failures.jsonl"
```

Only `role=candidate` rows run. The runner continues to refuse existing output files.

## 4. Generate a true sequential Stage 2

`--input-records` loads every successful Stage 1 `output_path` as the new edit input. It rejects
records that identify themselves as Stage 2, missing outputs, and duplicate source IDs.

```bash
python scripts/run_flux_kontext_probe.py \
  --input-records "$FACE_DESTYLE_ROOT/outputs/pair-bank-3d-v1/stage1/records.jsonl" \
  --required-style 3d_cartoon \
  --prompt-stage stage2 \
  --probe-stage batch \
  --model-dir "$FACE_DESTYLE_ROOT/models/diffusion/FLUX.1-Kontext-dev" \
  --download-manifest /path/to/download-manifest.json \
  --hash-manifest /path/to/hash-manifest.json \
  --output-dir "$FACE_DESTYLE_ROOT/outputs/pair-bank-3d-v1/stage2-sequential/images" \
  --records-output "$FACE_DESTYLE_ROOT/outputs/pair-bank-3d-v1/stage2-sequential/records.jsonl" \
  --failures-output "$FACE_DESTYLE_ROOT/outputs/pair-bank-3d-v1/stage2-sequential/failures.jsonl"
```

This differs from the historical material-extension Stage 2, which edited the original source a
second time.

## 5. Import teacher candidates and review

Place at most one optional closed-teacher image per source under
`closed-teacher/images/SOURCE_ID.png` (JPEG and WebP are also recognized). Record the teacher,
prompt, and date in `closed-teacher/NOTES.md`, then rerun `build_pair_bank_review.py`.

Review columns are exactly:

`styled source | FLUX Stage 1 | FLUX Stage 1 -> 2 | closed teacher`

Fill `selected_target` with `stage1`, `stage2-sequential`, or `closed-teacher`; set `decision` to
`accept` or `reject`. Select at most one target per source. Do not train a style-specific LoRA until
the operator has selected roughly 20--40 useful pairs for that style.
