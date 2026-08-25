#!/usr/bin/env python3
"""Build the 51-pair Origami V2.1 dataset with five concise prompt templates."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

EXPECTED_PAIRS = 51
EXPECTED_TEMPLATE_COUNTS = {
    "full_subject": 11,
    "hair_headwear": 10,
    "scalp_neck": 10,
    "clothing_support": 10,
    "identity_sensitive": 10,
}
PROTECTED_IDS = {
    "matv2-origami-002",
    "matv2-origami-007",
    "matv2-origami-011",
    "matv2-origami-018",
    "matv2-origami-023",
    "matv2-origami-030",
    "origami-hard-v2-021",
    "origami-hard-v2-023",
}

CORE_INSTRUCTION = (
    "Turn the entire folded-paper origami portrait into a natural realistic photograph. "
    "Replace all visible paper surfaces on the face, scalp, hair, headwear, ears, neck, "
    "clothing, shoulders, bust, pedestal, and support with realistic skin, hair, fabric, "
    "and materials. Preserve the same identity, apparent age, skin tone, facial hair, pose, "
    "gaze, expression, silhouette, crop, background, colors, and lighting."
)
TEMPLATES = {
    "full_subject": CORE_INSTRUCTION,
    "hair_headwear": (
        CORE_INSTRUCTION
        + " Fully convert the complete hair, hairline, ornaments, hood, scarf, or headwear; "
        "leave no paper folds around the head."
    ),
    "scalp_neck": (
        CORE_INSTRUCTION
        + " Fully convert the scalp, ears, jaw, and neck while preserving baldness and head "
        "shape; leave no planar paper facets in these regions."
    ),
    "clothing_support": (
        CORE_INSTRUCTION
        + " Fully convert the complete garment, shoulders, lower bust, pedestal, and support "
        "without cropping, deleting, or leaving folded-paper geometry."
    ),
    "identity_sensitive": (
        CORE_INSTRUCTION
        + " Where present, preserve wrinkles, gray or white facial hair, dark skin undertones, "
        "unusual gaze, profile, and dramatic shadow without rejuvenation or expression change."
    ),
}
TEMPLATE_ORDER = tuple(TEMPLATES)


def _load_metadata(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"metadata is missing: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid metadata JSON at line {line_number}") from exc
        rows.append(row)
    return rows


def _eligible_templates(tags: set[str]) -> list[str]:
    eligible = ["full_subject"]
    if tags & {"hair_headwear", "headscarf", "headwear"}:
        eligible.append("hair_headwear")
    if tags & {"scalp_neck", "bald", "geometric_exaggeration"}:
        eligible.append("scalp_neck")
    if tags & {"clothing_bust", "pedestal", "compound"}:
        eligible.append("clothing_support")
    if tags & {
        "beard",
        "elderly_wrinkles",
        "elderly",
        "dark_skin",
        "strong_shadow",
        "profile",
        "unusual_gaze",
    }:
        eligible.append("identity_sensitive")
    return eligible


def assign_templates(rows: list[dict]) -> list[dict]:
    """Assign valid templates with deterministic near-equal exposure."""
    counts: Counter[str] = Counter()
    rewritten = []
    for row in sorted(rows, key=lambda item: item["source_id"]):
        tags = set(row.get("difficulty_tags", []))
        eligible = _eligible_templates(tags)
        template = min(
            eligible,
            key=lambda name: (counts[name], TEMPLATE_ORDER.index(name)),
        )
        counts[template] += 1
        rewritten.append(
            {
                **row,
                "instruction": TEMPLATES[template],
                "instruction_template": template,
            }
        )
    if dict(counts) != EXPECTED_TEMPLATE_COUNTS:
        raise ValueError(f"unexpected template distribution: {dict(counts)}")
    return rewritten


def verify_dataset(root: Path) -> dict[str, int]:
    train = root / "train"
    rows = _load_metadata(train / "metadata.jsonl")
    if len(rows) != EXPECTED_PAIRS:
        raise ValueError(f"expected {EXPECTED_PAIRS} metadata rows, found {len(rows)}")
    source_ids = [str(row.get("source_id", "")) for row in rows]
    if len(set(source_ids)) != EXPECTED_PAIRS or any(not item for item in source_ids):
        raise ValueError("source IDs must be non-empty and unique")
    leaked = sorted(PROTECTED_IDS & set(source_ids))
    if leaked:
        raise ValueError("protected IDs leaked into training data: " + ", ".join(leaked))
    counts = Counter(str(row.get("instruction_template", "")) for row in rows)
    if dict(counts) != EXPECTED_TEMPLATE_COUNTS:
        raise ValueError(f"unexpected template distribution: {dict(counts)}")
    for row in rows:
        template = row["instruction_template"]
        if row.get("instruction") != TEMPLATES[template]:
            raise ValueError(f"instruction mismatch for {row['source_id']}")
        for field in ("condition_file_name", "target_file_name"):
            relative = Path(str(row.get(field, "")))
            if relative.is_absolute() or ".." in relative.parts or not relative.name:
                raise ValueError(f"non-portable {field} for {row['source_id']}")
            if not (train / relative).is_file():
                raise ValueError(f"missing {field} for {row['source_id']}: {relative}")
    condition_count = len(list((train / "condition").glob("*.png")))
    target_count = len(list((train / "target").glob("*.png")))
    if (condition_count, target_count) != (EXPECTED_PAIRS, EXPECTED_PAIRS):
        raise ValueError(
            f"expected 51 condition/target PNGs, found {condition_count}/{target_count}"
        )
    return dict(counts)


def build_dataset(source: Path, output: Path) -> dict[str, int]:
    if output.exists():
        raise ValueError(f"refusing to overwrite existing output: {output}")
    source_train = source / "train"
    rows = _load_metadata(source_train / "metadata.jsonl")
    if len(rows) != EXPECTED_PAIRS:
        raise ValueError(f"expected {EXPECTED_PAIRS} source rows, found {len(rows)}")
    rewritten = assign_templates(rows)
    output_train = output / "train"
    output_train.mkdir(parents=True)
    try:
        shutil.copytree(source_train / "condition", output_train / "condition")
        shutil.copytree(source_train / "target", output_train / "target")
        with (output_train / "metadata.jsonl").open("w", encoding="utf-8") as handle:
            for row in rewritten:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts = verify_dataset(output)
        (output / "BUILD_SUMMARY.json").write_text(
            json.dumps(
                {
                    "schema": "origami-lora-v21-template-dataset/v1",
                    "source_dataset": str(source.resolve()),
                    "pairs": EXPECTED_PAIRS,
                    "template_counts": counts,
                    "templates": TEMPLATES,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.verify_only:
            counts = verify_dataset(args.output)
        else:
            if args.source is None:
                parser.error("--source is required unless --verify-only is used")
            counts = build_dataset(args.source, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"DATASET_OK={EXPECTED_PAIRS} TEMPLATE_COUNTS={json.dumps(counts, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
