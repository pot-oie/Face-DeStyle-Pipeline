#!/usr/bin/env python3
"""Build the exact 51-pair Origami LoRA V2 ImageFolder dataset."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import NamedTuple

from PIL import Image, ImageDraw, ImageOps

EXPECTED_V1_COUNT = 23
EXPECTED_V2_COUNT = 28
EXPECTED_TOTAL_COUNT = 51
REVIEW_EXCLUSIONS = {"origami-hard-v2-021", "origami-hard-v2-023"}
PROTECTED_HOLDOUTS = {
    "matv2-origami-002",
    "matv2-origami-007",
    "matv2-origami-011",
    "matv2-origami-018",
    "matv2-origami-023",
    "matv2-origami-030",
}

BASE_INSTRUCTION = (
    "Convert the complete folded-paper origami subject into a natural photorealistic camera "
    "portrait. Replace every visible folded-paper surface across the skin, scalp, hair, "
    "headwear, eyebrows, eyelashes, ears, beard, moustache, neck, clothing, shoulders, bust, "
    "and pedestal with biologically plausible human features, individual hair strands, real "
    "accessories, naturally draped woven fabric, and a physically plausible support where "
    "present. Preserve the same fictional identity, apparent age, skin tone, body type, facial "
    "proportions, pose, gaze, expression, silhouette, hairstyle or baldness, accessory shapes, "
    "crop, garment layout, background, palette, and lighting."
)

TAG_CLAUSES = {
    "hair_headwear": (
        "Fully naturalize the hair, hairline, hair ornaments, hood, scarf, or other headwear; "
        "use individual hair fibers or clearly woven textile without folded edges."
    ),
    "beard": (
        "Turn the beard and moustache into individual natural fibers while preserving their "
        "shape, density, color, and the mouth beneath them."
    ),
    "elderly_wrinkles": (
        "Render wrinkles and age-related skin anatomy naturally without smoothing, beautifying, "
        "or rejuvenating the subject."
    ),
    "scalp_neck": (
        "Remove paper planes from the scalp, ears, jaw, and neck while preserving baldness, head "
        "shape, ear shape, and neck proportions."
    ),
    "clothing_bust": (
        "Convert the complete garment, shoulders, lower bust, and any visible pedestal or support "
        "to woven fabric and physically plausible materials without cropping or removing them."
    ),
    "geometric_exaggeration": (
        "Replace planar facets with plausible anatomy while retaining the depicted facial "
        "proportions, silhouette, and deliberate shape characteristics."
    ),
    "profile": "Keep the exact side or three-quarter pose and visible facial outline.",
    "unusual_gaze": "Keep the exact eye direction and unusual gaze.",
    "dark_skin": "Preserve the subject's dark skin tone and undertone exactly.",
    "strong_shadow": "Preserve the direction, contrast, and placement of the dramatic lighting.",
}

TAG_ALIASES = {
    "bald": "scalp_neck",
    "elderly": "elderly_wrinkles",
    "headscarf": "hair_headwear",
    "headwear": "hair_headwear",
    "pedestal": "clothing_bust",
}


class Pair(NamedTuple):
    source_id: str
    condition: Path
    target: Path
    difficulty_tags: tuple[str, ...]
    reviewer_notes: str
    pair_source: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _portable_path(value: str, *, field: str, source_id: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{source_id} has non-portable {field}: {value}")
    if not value or path.name == "":
        raise ValueError(f"{source_id} has empty {field}")
    return path


def load_v2_selection(path: Path, v2_root: Path) -> list[Pair]:
    rows = _read_csv(path)
    required = {
        "source_id",
        "condition_path",
        "target_path",
        "decision",
        "difficulty_tags",
        "reviewer_notes",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("V2 selection CSV lacks required columns")
    source_ids = [row["source_id"] for row in rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("V2 selection CSV contains duplicate source IDs")
    decisions = {row["source_id"]: row["decision"] for row in rows}
    if not REVIEW_EXCLUSIONS.issubset(decisions):
        raise ValueError("V2 selection CSV omits an independently reviewed exclusion")
    incorrectly_accepted = sorted(
        source_id
        for source_id in REVIEW_EXCLUSIONS
        if decisions[source_id] == "accept"
    )
    if incorrectly_accepted:
        raise ValueError("review exclusions were accepted: " + ", ".join(incorrectly_accepted))

    accepted: list[Pair] = []
    for row in rows:
        if row["decision"] != "accept":
            continue
        source_id = row["source_id"]
        condition_relative = _portable_path(
            row["condition_path"], field="condition_path", source_id=source_id
        )
        target_relative = _portable_path(
            row["target_path"], field="target_path", source_id=source_id
        )
        if condition_relative.stem != source_id or target_relative.stem != source_id:
            raise ValueError(f"V2 source/target filename mismatch for {source_id}")
        tags = tuple(tag.strip() for tag in row["difficulty_tags"].split(",") if tag.strip())
        if not tags:
            raise ValueError(f"V2 pair lacks difficulty tags: {source_id}")
        accepted.append(
            Pair(
                source_id=source_id,
                condition=v2_root / condition_relative,
                target=v2_root / target_relative,
                difficulty_tags=tags,
                reviewer_notes=row["reviewer_notes"],
                pair_source="hard-pair-v2",
            )
        )
    if len(accepted) != EXPECTED_V2_COUNT:
        raise ValueError(f"expected {EXPECTED_V2_COUNT} accepted V2 pairs, found {len(accepted)}")
    return accepted


def infer_v1_tags(reviewer_notes: str) -> tuple[str, ...]:
    text = reviewer_notes.lower()
    tags: list[str] = []
    keywords = {
        "hair_headwear": ("hair", "bun", "curl", "hood", "headband", "head covering"),
        "beard": ("beard", "moustache"),
        "elderly_wrinkles": ("aged", "age", "elderly", "mature", "older"),
        "scalp_neck": ("scalp", "bald", "ear", "neck"),
        "clothing_bust": ("clothing", "garment", "coat", "robe", "bust", "pedestal"),
        "unusual_gaze": ("sideways gaze", "upward gaze", "downward gaze"),
    }
    for tag, needles in keywords.items():
        if any(needle in text for needle in needles):
            tags.append(tag)
    if not tags:
        tags.append("clothing_bust")
    return tuple(tags)


def load_v1_pairs(v1_root: Path) -> list[Pair]:
    metadata_path = v1_root / "train/metadata.jsonl"
    metadata = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(metadata) != EXPECTED_V1_COUNT:
        raise ValueError(f"expected {EXPECTED_V1_COUNT} V1 pairs, found {len(metadata)}")
    notes_rows = _read_csv(v1_root / "target_selection.csv")
    notes = {
        row["source_id"]: row.get("reviewer_notes", "")
        for row in notes_rows
        if row.get("decision") == "accept"
    }
    pairs: list[Pair] = []
    for row in metadata:
        source_id = row["source_id"]
        condition_relative = _portable_path(
            row["condition_file_name"], field="condition_file_name", source_id=source_id
        )
        target_relative = _portable_path(
            row["target_file_name"], field="target_file_name", source_id=source_id
        )
        if condition_relative.stem != source_id or target_relative.stem != source_id:
            raise ValueError(f"V1 source/target filename mismatch for {source_id}")
        if source_id not in notes:
            raise ValueError(f"V1 accepted selection note missing for {source_id}")
        pairs.append(
            Pair(
                source_id=source_id,
                condition=v1_root / "train" / condition_relative,
                target=v1_root / "train" / target_relative,
                difficulty_tags=infer_v1_tags(notes[source_id]),
                reviewer_notes=notes[source_id],
                pair_source="strict-v1",
            )
        )
    source_ids = [pair.source_id for pair in pairs]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("V1 metadata contains duplicate source IDs")
    return pairs


def normalized_tags(tags: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for tag in tags:
        canonical = TAG_ALIASES.get(tag, tag)
        if canonical == "compound" or canonical in normalized:
            continue
        normalized.append(canonical)
    return tuple(normalized)


def build_instruction(tags: tuple[str, ...]) -> str:
    clauses = [TAG_CLAUSES[tag] for tag in normalized_tags(tags) if tag in TAG_CLAUSES]
    if not clauses:
        raise ValueError("difficulty tags do not map to any regional instruction clause")
    return " ".join([BASE_INSTRUCTION, *clauses])


def _validate_images(pairs: list[Pair]) -> None:
    for pair in pairs:
        for role, path in (("condition", pair.condition), ("target", pair.target)):
            if not path.is_file():
                raise FileNotFoundError(f"missing {role} for {pair.source_id}: {path}")
            with Image.open(path) as image:
                image.verify()


def _render_contact_page(pairs: list[Pair], destination: Path) -> None:
    panel = 220
    label = 26
    header = 30
    gap = 8
    columns = 2
    rows = (len(pairs) + columns - 1) // columns
    pair_width = panel * 2 + gap
    width = columns * pair_width + (columns + 1) * gap
    height = header + rows * (panel + label + gap) + gap
    canvas = Image.new("RGB", (width, height), (248, 248, 246))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 8), "STYLED CONDITION  |  NATURAL TARGET", fill=(20, 20, 20))
    for index, pair in enumerate(pairs):
        column = index % columns
        row = index // columns
        left = gap + column * (pair_width + gap)
        top = header + row * (panel + label + gap)
        for offset, path in ((0, pair.condition), (panel + gap, pair.target)):
            with Image.open(path) as image:
                rendered = ImageOps.pad(
                    ImageOps.exif_transpose(image).convert("RGB"),
                    (panel, panel),
                    method=Image.Resampling.LANCZOS,
                    color=(232, 232, 232),
                )
            canvas.paste(rendered, (left + offset, top))
        draw.text((left + 3, top + panel + 5), pair.source_id, fill=(20, 20, 20))
    canvas.save(destination, format="JPEG", quality=92, optimize=True)


def render_contact_sheets(pairs: list[Pair], output_dir: Path) -> None:
    contact_dir = output_dir / "contact-sheets"
    contact_dir.mkdir()
    page_size = 12
    for start in range(0, len(pairs), page_size):
        page = start // page_size + 1
        _render_contact_page(pairs[start : start + page_size], contact_dir / f"page-{page:02d}.jpg")
    _render_contact_page(pairs, output_dir / "preview.jpg")


def _save_rgb(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        ImageOps.exif_transpose(image).convert("RGB").save(destination)


def build_dataset(*, v1_root: Path, v2_root: Path, selection: Path, output_dir: Path) -> int:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    pairs = load_v1_pairs(v1_root) + load_v2_selection(selection, v2_root)
    source_ids = [pair.source_id for pair in pairs]
    if len(pairs) != EXPECTED_TOTAL_COUNT or len(source_ids) != len(set(source_ids)):
        raise ValueError("combined dataset is not exactly 51 unique pairs")
    protected = sorted(PROTECTED_HOLDOUTS.intersection(source_ids))
    if protected:
        raise ValueError("protected holdouts entered training data: " + ", ".join(protected))
    _validate_images(pairs)

    train_dir = output_dir / "train"
    condition_dir = train_dir / "condition"
    target_dir = train_dir / "target"
    condition_dir.mkdir(parents=True)
    target_dir.mkdir()
    metadata: list[dict[str, str]] = []
    for pair in pairs:
        name = f"{pair.source_id}.png"
        _save_rgb(pair.condition, condition_dir / name)
        _save_rgb(pair.target, target_dir / name)
        metadata.append(
            {
                "target_file_name": f"target/{name}",
                "condition_file_name": f"condition/{name}",
                "instruction": build_instruction(pair.difficulty_tags),
                "source_id": pair.source_id,
                "pair_source": pair.pair_source,
                "difficulty_tags": list(pair.difficulty_tags),
                "reviewer_notes": pair.reviewer_notes,
            }
        )
    (train_dir / "metadata.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metadata),
        encoding="utf-8",
    )
    shutil.copy2(selection, output_dir / "origami_hard_v2_selection.csv")
    (output_dir / "BUILD_SUMMARY.json").write_text(
        json.dumps(
            {
                "schema": "origami-lora-imagefolder/v2",
                "pair_count": EXPECTED_TOTAL_COUNT,
                "strict_v1_pairs": EXPECTED_V1_COUNT,
                "hard_v2_pairs": EXPECTED_V2_COUNT,
                "excluded_hard_v2_ids": sorted(REVIEW_EXCLUSIONS),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    render_contact_sheets(pairs, output_dir)
    return len(metadata)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-root", type=Path, required=True)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path(
            "data/manifests/multistyle-pair-bank/origami_hard_v2_selection.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        count = build_dataset(
            v1_root=args.v1_root,
            v2_root=args.v2_root,
            selection=args.selection,
            output_dir=args.output_dir,
        )
    except (FileExistsError, FileNotFoundError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Built {count} Origami V2 pairs in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
