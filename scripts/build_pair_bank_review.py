#!/usr/bin/env python3
"""Prepare pair-bank directories, contact sheets, and four-column review sheets."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from face_destyle.data.pair_bank import (
    PAIR_BANK_ROLES,
    PairBankSource,
    load_pair_bank_source_list,
)

CONTACT_SIZE = 240
CONTACT_LABEL_HEIGHT = 48
REVIEW_SIZE = 320
REVIEW_HEADER_HEIGHT = 42
REVIEW_ROW_LABEL_HEIGHT = 30
GAP = 8


def image_panel(path: Path, size: int) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.pad(
            ImageOps.exif_transpose(image).convert("RGB"),
            (size, size),
            method=Image.Resampling.LANCZOS,
            color=(232, 232, 232),
        )


def placeholder(size: int, label: str) -> Image.Image:
    panel = Image.new("RGB", (size, size), (235, 235, 235))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((1, 1, size - 2, size - 2), outline=(180, 180, 180), width=2)
    draw.text((12, size // 2 - 8), label, fill=(90, 90, 90))
    return panel


def build_contact_sheet(
    rows: list[PairBankSource], destination: Path, columns: int = 5
) -> None:
    count_rows = math.ceil(len(rows) / columns)
    width = columns * CONTACT_SIZE + (columns - 1) * GAP
    height = count_rows * (CONTACT_SIZE + CONTACT_LABEL_HEIGHT + GAP) - GAP
    canvas = Image.new("RGB", (width, height), (248, 248, 246))
    draw = ImageDraw.Draw(canvas)
    role_colors = {
        "candidate": (20, 115, 55),
        "holdout": (40, 80, 170),
        "rejected": (155, 45, 45),
    }
    for index, row in enumerate(rows):
        grid_row, column = divmod(index, columns)
        left = column * (CONTACT_SIZE + GAP)
        top = grid_row * (CONTACT_SIZE + CONTACT_LABEL_HEIGHT + GAP)
        canvas.paste(image_panel(row.image_path, CONTACT_SIZE), (left, top))
        draw.text((left + 4, top + CONTACT_SIZE + 4), row.source_id, fill=(20, 20, 20))
        draw.text(
            (left + 4, top + CONTACT_SIZE + 22),
            row.role,
            fill=role_colors[row.role],
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="JPEG", quality=90, optimize=True)


def candidate_path(directory: Path, source_id: str) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = directory / f"{source_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def build_review_pages(
    rows: list[PairBankSource], run_dir: Path, rows_per_page: int = 6
) -> int:
    review_rows = [row for row in rows if row.role == "candidate"]
    columns = (
        ("STYLED SOURCE", None),
        ("FLUX STAGE 1", run_dir / "stage1" / "images"),
        ("FLUX STAGE 1 -> 2", run_dir / "stage2-sequential" / "images"),
        ("CLOSED TEACHER", run_dir / "closed-teacher" / "images"),
    )
    page_count = math.ceil(len(review_rows) / rows_per_page)
    sheets_dir = run_dir / "review" / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    for page_index in range(page_count):
        page_rows = review_rows[
            page_index * rows_per_page : (page_index + 1) * rows_per_page
        ]
        width = len(columns) * REVIEW_SIZE + (len(columns) - 1) * GAP
        row_height = REVIEW_SIZE + REVIEW_ROW_LABEL_HEIGHT + GAP
        height = REVIEW_HEADER_HEIGHT + len(page_rows) * row_height - GAP
        canvas = Image.new("RGB", (width, height), (248, 248, 246))
        draw = ImageDraw.Draw(canvas)
        for column_index, (label, _directory) in enumerate(columns):
            left = column_index * (REVIEW_SIZE + GAP)
            draw.text((left + 8, 14), label, fill=(20, 20, 20))
        for row_index, row in enumerate(page_rows):
            top = REVIEW_HEADER_HEIGHT + row_index * row_height
            for column_index, (_label, directory) in enumerate(columns):
                left = column_index * (REVIEW_SIZE + GAP)
                path = row.image_path if directory is None else candidate_path(
                    directory, row.source_id
                )
                panel = (
                    image_panel(path, REVIEW_SIZE)
                    if path is not None
                    else placeholder(REVIEW_SIZE, "MISSING")
                )
                canvas.paste(panel, (left, top))
            draw.text(
                (4, top + REVIEW_SIZE + 7),
                row.source_id,
                fill=(20, 20, 20),
            )
        canvas.save(
            sheets_dir / f"review-{page_index + 1:02d}.jpg",
            format="JPEG",
            quality=90,
            optimize=True,
        )
    return page_count


def prepare_layout(run_dir: Path, rows: list[PairBankSource]) -> None:
    for relative in (
        "stage1/images",
        "stage2-sequential/images",
        "closed-teacher/images",
        "review/sheets",
    ):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    teacher_notes = run_dir / "closed-teacher" / "NOTES.md"
    if not teacher_notes.exists():
        teacher_notes.write_text(
            "# Closed-teacher candidates\n\n"
            "Place one optional image per source in `images/SOURCE_ID.png` (JPEG/WebP also work). "
            "Record the teacher model, prompt, and date here.\n",
            encoding="utf-8",
        )
    selection = run_dir / "review" / "target_selection.csv"
    if not selection.exists():
        with selection.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "source_id",
                    "selected_target",
                    "decision",
                    "reviewer_notes",
                ),
            )
            writer.writeheader()
            for row in rows:
                if row.role == "candidate":
                    writer.writerow(
                        {
                            "source_id": row.source_id,
                            "selected_target": "",
                            "decision": "",
                            "reviewer_notes": "",
                        }
                    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-list", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contact-columns", type=int, default=5)
    parser.add_argument("--review-rows-per-page", type=int, default=6)
    args = parser.parse_args()
    if args.contact_columns < 1 or args.review_rows_per_page < 1:
        parser.error("sheet dimensions must be positive")
    try:
        rows = load_pair_bank_source_list(args.source_list, args.data_root)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    prepare_layout(args.run_dir, rows)
    build_contact_sheet(
        rows,
        args.run_dir / "review" / "3d-source-inventory.jpg",
        columns=args.contact_columns,
    )
    page_count = build_review_pages(
        rows,
        args.run_dir,
        rows_per_page=args.review_rows_per_page,
    )
    counts = {role: sum(row.role == role for row in rows) for role in PAIR_BANK_ROLES}
    print(
        f"Prepared {len(rows)} sources ({counts['candidate']} candidate, "
        f"{counts['holdout']} holdout, {counts['rejected']} rejected) and "
        f"{page_count} review sheets in {args.run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
