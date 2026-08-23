#!/usr/bin/env python3
"""Build a FLUX Kontext ImageFolder dataset from reviewed pair-bank targets."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from face_destyle.data.pair_bank import load_pair_bank_source_list
from face_destyle.filtering.prompt_rewriter import select_stage_prompt
from face_destyle.utils.io import load_yaml

TARGET_DIRS = {
    "stage1": Path("stage1/images"),
    "stage2-sequential": Path("stage2-sequential/images"),
    "closed-teacher": Path("closed-teacher/images"),
}


def load_accepted_selection(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"source_id", "selected_target", "decision"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("selection CSV lacks required columns")
    accepted = [row for row in rows if row["decision"] == "accept"]
    if not accepted:
        raise ValueError("selection CSV contains no accepted targets")
    source_ids = [row["source_id"] for row in accepted]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("selection CSV contains duplicate accepted source IDs")
    invalid = sorted(
        {
            row["selected_target"]
            for row in accepted
            if row["selected_target"] not in TARGET_DIRS
        }
    )
    if invalid:
        raise ValueError("unknown selected target: " + ", ".join(invalid))
    return accepted


def find_target(run_dir: Path, source_id: str, selected_target: str) -> Path:
    directory = run_dir / TARGET_DIRS[selected_target]
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        path = directory / f"{source_id}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"missing {selected_target} target for {source_id}: {directory}"
    )


def render_preview(pairs: list[tuple[str, Path, Path]], destination: Path) -> None:
    panel_size = 192
    label_height = 26
    gap = 8
    rows_per_column = 5
    columns = (len(pairs) + rows_per_column - 1) // rows_per_column
    pair_width = panel_size * 2 + gap
    width = columns * pair_width + (columns - 1) * (gap * 2)
    height = rows_per_column * (panel_size + label_height + gap) - gap
    canvas = Image.new("RGB", (width, height), (248, 248, 246))
    draw = ImageDraw.Draw(canvas)
    for index, (source_id, condition, target) in enumerate(pairs):
        column, row = divmod(index, rows_per_column)
        left = column * (pair_width + gap * 2)
        top = row * (panel_size + label_height + gap)
        for offset, path in ((0, condition), (panel_size + gap, target)):
            with Image.open(path) as image:
                panel = ImageOps.pad(
                    ImageOps.exif_transpose(image).convert("RGB"),
                    (panel_size, panel_size),
                    method=Image.Resampling.LANCZOS,
                    color=(232, 232, 232),
                )
            canvas.paste(panel, (left + offset, top))
        draw.text((left + 4, top + panel_size + 5), source_id, fill=(20, 20, 20))
    canvas.save(destination, format="JPEG", quality=92, optimize=True)


def build_dataset(
    *,
    source_list: Path,
    selection: Path,
    data_root: Path,
    run_dir: Path,
    output_dir: Path,
    styles_config: Path,
    style_category: str,
) -> int:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    sources = {
        row.source_id: row
        for row in load_pair_bank_source_list(
            source_list, data_root, roles={"candidate"}
        )
    }
    accepted = load_accepted_selection(selection)
    selected_ids = {row["source_id"] for row in accepted}
    missing_sources = sorted(selected_ids - sources.keys())
    if missing_sources:
        raise ValueError(
            "accepted source IDs are absent from source list: "
            + ", ".join(missing_sources)
        )
    wrong_styles = sorted(
        source_id
        for source_id in selected_ids
        if sources[source_id].style_category != style_category
    )
    if wrong_styles:
        raise ValueError("accepted sources have wrong style: " + ", ".join(wrong_styles))

    config = load_yaml(styles_config)
    instruction = select_stage_prompt(style_category, config, stage="stage1")
    target_paths = {
        row["source_id"]: find_target(
            run_dir, row["source_id"], row["selected_target"]
        )
        for row in accepted
    }

    train_dir = output_dir / "train"
    condition_dir = train_dir / "condition"
    target_dir = train_dir / "target"
    condition_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)
    metadata: list[dict[str, str]] = []
    preview_pairs: list[tuple[str, Path, Path]] = []
    for row in accepted:
        source_id = row["source_id"]
        name = f"{source_id}.png"
        condition_destination = condition_dir / name
        target_destination = target_dir / name
        with Image.open(sources[source_id].image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.save(condition_destination)
        with Image.open(target_paths[source_id]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.save(target_destination)
        metadata.append(
            {
                "target_file_name": f"target/{name}",
                "condition_file_name": f"condition/{name}",
                "instruction": instruction,
                "source_id": source_id,
                "selected_target": row["selected_target"],
            }
        )
        preview_pairs.append(
            (source_id, condition_destination, target_destination)
        )
    (train_dir / "metadata.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metadata),
        encoding="utf-8",
    )
    shutil.copy2(selection, output_dir / "target_selection.csv")
    render_preview(preview_pairs, output_dir / "preview.jpg")
    return len(metadata)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-list", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--styles-config", type=Path, default=Path("configs/styles.yaml"))
    parser.add_argument("--style-category", required=True)
    args = parser.parse_args()
    try:
        count = build_dataset(
            source_list=args.source_list,
            selection=args.selection,
            data_root=args.data_root,
            run_dir=args.run_dir,
            output_dir=args.output_dir,
            styles_config=args.styles_config,
            style_category=args.style_category,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Built {count} accepted pairs in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
