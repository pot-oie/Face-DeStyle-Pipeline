#!/usr/bin/env python3
"""Build the frozen formal-v1 300-pair primary review and 60-pair retest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from face_destyle.data.manifests import load_dataset_manifest
from face_destyle.data.metadata import read_jsonl
from face_destyle.schemas import DestylizationRecord

METHOD_COUNT = 5
SOURCE_COUNT = 60
STYLES = ("3d_cartoon", "comic", "ink", "watercolor")
SOURCES_PER_STYLE = 15
REPEATS_PER_CELL = 3
PRIMARY_SEED = 20260821
REPEAT_SEED = 20260822
PANEL_SIZE = 768
HEADER_HEIGHT = 44
GAP = 12


def parse_records(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("records must use METHOD=PATH")
    method, raw_path = value.split("=", 1)
    if not method.strip() or not raw_path.strip():
        raise ValueError("records must use METHOD=PATH")
    return method.strip(), Path(raw_path).expanduser().resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_candidates(
    manifest: Path,
    data_root: Path,
    method_records: list[tuple[str, Path]],
) -> tuple[list[dict[str, str]], list[str]]:
    methods = [method for method, _path in method_records]
    if len(methods) != METHOD_COUNT or len(set(methods)) != METHOD_COUNT:
        raise ValueError(f"formal-v1 requires exactly {METHOD_COUNT} unique methods")
    sources = load_dataset_manifest(manifest, data_root=data_root, split="test")
    if len(sources) != SOURCE_COUNT:
        raise ValueError(f"formal-v1 requires exactly {SOURCE_COUNT} test sources")
    style_counts = Counter(source.style_category for source in sources)
    expected_style_counts = {style: SOURCES_PER_STYLE for style in STYLES}
    if dict(style_counts) != expected_style_counts:
        raise ValueError(
            f"test styles must equal {expected_style_counts}; received {dict(style_counts)}"
        )
    expected = {source.source_id: source for source in sources}
    candidates: list[dict[str, str]] = []
    for method, records_path in method_records:
        records = read_jsonl(records_path, DestylizationRecord)
        selected: dict[str, DestylizationRecord] = {}
        for record in records:
            if record.source_id not in expected:
                continue
            if record.source_id in selected:
                raise ValueError(f"duplicate test source ID in {method}: {record.source_id}")
            selected[record.source_id] = record
        missing = sorted(set(expected) - set(selected))
        if missing:
            raise ValueError(f"{method} lacks {len(missing)} test records; first={missing[0]}")
        for source_id in sorted(expected):
            source = expected[source_id]
            record = selected[source_id]
            source_asset = Path(source.image_path).resolve().relative_to(data_root.resolve())
            if not Path(record.input_path).as_posix().endswith(source_asset.as_posix()):
                raise ValueError(f"input mismatch for {method}/{source_id}")
            if record.style_category != source.style_category:
                raise ValueError(f"style mismatch for {method}/{source_id}")
            if record.seed != 42:
                raise ValueError(f"seed mismatch for {method}/{source_id}: {record.seed}")
            declared_output = Path(record.output_path).expanduser()
            output_candidates = (
                declared_output,
                records_path.parent / "images" / declared_output.name,
                records_path.parent / declared_output.name,
            )
            candidate_path = next(
                (path.resolve() for path in output_candidates if path.is_file()), None
            )
            if candidate_path is None:
                raise ValueError(
                    f"missing candidate for {method}/{source_id}; declared={record.output_path}"
                )
            candidates.append(
                {
                    "canonical_id": f"{method}:{source_id}",
                    "method": method,
                    "source_id": source_id,
                    "style_category": source.style_category,
                    "source_path": str(Path(source.image_path).resolve()),
                    "candidate_path": str(candidate_path),
                }
            )
    if len(candidates) != METHOD_COUNT * SOURCE_COUNT:
        raise AssertionError("candidate matrix is incomplete")
    output_paths = [row["candidate_path"] for row in candidates]
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("a candidate output path is reused across method-source pairs")
    return candidates, methods


def select_repeat(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    cells: defaultdict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for candidate in candidates:
        cells[(candidate["method"], candidate["style_category"])].append(candidate)
    if len(cells) != METHOD_COUNT * len(STYLES):
        raise ValueError("repeat strata are incomplete")
    rng = random.Random(REPEAT_SEED)
    selected: list[dict[str, str]] = []
    for cell in sorted(cells):
        members = sorted(cells[cell], key=lambda row: row["canonical_id"])
        if len(members) != SOURCES_PER_STYLE:
            raise ValueError(f"repeat stratum {cell} has {len(members)} candidates, expected 15")
        selected.extend(rng.sample(members, REPEATS_PER_CELL))
    rng.shuffle(selected)
    return selected


def render_pair(source: Path, candidate: Path, destination: Path) -> None:
    canvas = Image.new(
        "RGB", (PANEL_SIZE * 2 + GAP, PANEL_SIZE + HEADER_HEIGHT), (245, 245, 245)
    )
    draw = ImageDraw.Draw(canvas)
    for index, (path, label) in enumerate(((source, "SOURCE"), (candidate, "CANDIDATE"))):
        with Image.open(path) as image:
            panel = ImageOps.pad(
                ImageOps.exif_transpose(image).convert("RGB"),
                (PANEL_SIZE, PANEL_SIZE),
                method=Image.Resampling.LANCZOS,
                color=(232, 232, 232),
            )
        left = index * (PANEL_SIZE + GAP)
        canvas.paste(panel, (left, HEADER_HEIGHT))
        draw.text((left + 12, 14), label, fill=(20, 20, 20))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="JPEG", quality=92, optimize=True)


def write_instructions(path: Path) -> None:
    path.write_text(
        """# Formal-v1 held-out blinded review

Score SOURCE and CANDIDATE at equal displayed size. Method, source ID, filenames, generation
metadata, and automatic metrics are hidden. Use integer scores from 0 to 5.

- Content: preservation of subject, pose, composition, objects, and spatial relations.
- Style removal: 5 is photographic with no meaningful residual source style; 0 is failed.
- Recoverable identity: preservation of visible identity-bearing facial evidence.
- Set `identity_judgment_valid` to `no` when identity cannot be judged. This candidate fails.
- `failure_types` may be blank. Blank means `not_reported`, never "no failure".

A pass requires content >= 4, style removal >= 4, identity judgment valid = yes, and identity >= 4.
Missing core scores are not imputed and invalidate the candidate. Freeze all primary scores before
opening the delayed repeat directory. Keep `private/private_key.jsonl` closed until both score files
are frozen; do not adjudicate after unblinding.
""",
        encoding="utf-8",
    )


def write_round(
    root: Path,
    round_name: str,
    prefix: str,
    ordered: list[dict[str, str]],
    private_rows: list[dict[str, str]],
) -> Path:
    columns = [
        "blind_id",
        "style_category",
        "content_score",
        "style_removal_score",
        "identity_score",
        "identity_judgment_valid",
        "failure_types",
        "reviewer_notes",
    ]
    rows = []
    for index, candidate in enumerate(ordered, start=1):
        blind_id = f"{prefix}-{index:04d}"
        pair_path = Path("pairs") / f"{blind_id}.jpg"
        render_pair(
            Path(candidate["source_path"]),
            Path(candidate["candidate_path"]),
            root / pair_path,
        )
        rows.append(
            {
                "blind_id": blind_id,
                "style_category": candidate["style_category"],
                "content_score": "",
                "style_removal_score": "",
                "identity_score": "",
                "identity_judgment_valid": "",
                "failure_types": "",
                "reviewer_notes": "",
            }
        )
        private_rows.append(
            {
                "round": round_name,
                "blind_id": blind_id,
                "pair_path": str(Path(round_name) / pair_path),
                **candidate,
            }
        )
    scores = root / "scores.csv"
    with scores.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--records",
        action="append",
        required=True,
        metavar="METHOD=PATH",
        help="Method label and complete generation records; repeat exactly five times.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"refusing non-empty output directory: {args.output_dir}")
    try:
        candidates, methods = collect_candidates(
            args.manifest,
            args.data_root,
            [parse_records(value) for value in args.records],
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    primary = sorted(candidates, key=lambda row: row["canonical_id"])
    random.Random(PRIMARY_SEED).shuffle(primary)
    repeat = select_repeat(candidates)
    reviewer = args.output_dir / "reviewer"
    private = args.output_dir / "private"
    reviewer.mkdir(parents=True)
    private.mkdir(parents=True)
    write_instructions(reviewer / "INSTRUCTIONS.md")
    private_rows: list[dict[str, str]] = []
    primary_scores = write_round(
        reviewer / "primary", "primary", "P", primary, private_rows
    )
    repeat_scores = write_round(
        reviewer / "repeat", "repeat", "R", repeat, private_rows
    )
    key_path = private / "private_key.jsonl"
    with key_path.open("w", encoding="utf-8") as handle:
        for row in private_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    repeat_counts = Counter((row["method"], row["style_category"]) for row in repeat)
    build = {
        "schema": "face-destyle-formal-v1-heldout-blind-review/v1",
        "primary_seed": PRIMARY_SEED,
        "repeat_seed": REPEAT_SEED,
        "method_count": METHOD_COUNT,
        "source_count": SOURCE_COUNT,
        "primary_candidate_count": len(primary),
        "repeat_candidate_count": len(repeat),
        "repeat_per_method_style_cell": REPEATS_PER_CELL,
        "methods": methods,
        "repeat_cell_counts": {
            f"{method}|{style}": count
            for (method, style), count in sorted(repeat_counts.items())
        },
        "primary_scores_sha256": file_sha256(primary_scores),
        "repeat_scores_sha256": file_sha256(repeat_scores),
        "private_key_sha256": file_sha256(key_path),
    }
    (private / "build.json").write_text(
        json.dumps(build, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Built {len(primary)} primary pairs and {len(repeat)} stratified repeat pairs "
        f"in {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
