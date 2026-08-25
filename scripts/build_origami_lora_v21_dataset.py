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
    "Make the entire origami portrait a natural photo. Remove paper from visible skin, scalp, "
    "hair, headwear, neck, clothes, shoulders, bust, pedestal and support. Keep identity, age, "
    "skin tone, facial hair, pose, gaze, expression, composition, background and lighting."
)
TEMPLATES = {
    "full_subject": CORE_INSTRUCTION,
    "hair_headwear": (
        CORE_INSTRUCTION
        + " Naturalize all hair and headwear; leave no paper folds."
    ),
    "scalp_neck": (
        CORE_INSTRUCTION
        + " Keep baldness; naturalize scalp and neck without adding hair."
    ),
    "clothing_support": (
        CORE_INSTRUCTION
        + " Naturalize complete clothing, bust, pedestal and support without cropping."
    ),
    "identity_sensitive": (
        CORE_INSTRUCTION
        + " Preserve wrinkles, gray beard, dark skin, gaze, shadows and expression."
    ),
}
TEMPLATE_ORDER = tuple(TEMPLATES)
CLIP_TOKEN_LIMIT = 77
CONSERVATIVE_WORD_LIMIT = 50


def validate_conservative_prompt_lengths() -> None:
    """Keep every template short even before the exact tokenizer is available."""
    too_long = {
        name: len(prompt.split())
        for name, prompt in TEMPLATES.items()
        if len(prompt.split()) > CONSERVATIVE_WORD_LIMIT
    }
    if too_long:
        raise ValueError(f"prompt templates exceed conservative word limit: {too_long}")


def validate_clip_token_lengths(tokenizer: object) -> dict[str, int]:
    """Reject any template that the model's CLIP tokenizer would truncate."""
    counts = {}
    for name, prompt in TEMPLATES.items():
        encoded = tokenizer(prompt, add_special_tokens=True, truncation=False)
        input_ids = encoded["input_ids"]
        counts[name] = len(input_ids)
    too_long = {name: count for name, count in counts.items() if count > CLIP_TOKEN_LIMIT}
    if too_long:
        raise ValueError(f"CLIP prompt templates exceed {CLIP_TOKEN_LIMIT} tokens: {too_long}")
    return counts


def load_and_validate_clip_tokenizer(path: Path) -> dict[str, int]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ValueError("transformers is required for exact CLIP token validation") from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    except (OSError, ValueError) as exc:
        raise ValueError(f"failed to load local CLIP tokenizer: {path}") from exc
    return validate_clip_token_lengths(tokenizer)


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


def verify_dataset(
    root: Path, *, clip_tokenizer: Path | None = None
) -> tuple[dict[str, int], dict[str, int] | None]:
    validate_conservative_prompt_lengths()
    clip_token_counts = (
        load_and_validate_clip_tokenizer(clip_tokenizer)
        if clip_tokenizer is not None
        else None
    )
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
    return dict(counts), clip_token_counts


def build_dataset(
    source: Path, output: Path, *, clip_tokenizer: Path | None = None
) -> tuple[dict[str, int], dict[str, int] | None]:
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
        counts, clip_token_counts = verify_dataset(
            output, clip_tokenizer=clip_tokenizer
        )
        (output / "BUILD_SUMMARY.json").write_text(
            json.dumps(
                {
                    "schema": "origami-lora-v21-template-dataset/v1",
                    "source_dataset": str(source.resolve()),
                    "pairs": EXPECTED_PAIRS,
                    "template_counts": counts,
                    "clip_token_limit": CLIP_TOKEN_LIMIT,
                    "clip_token_counts": clip_token_counts,
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
    return counts, clip_token_counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--clip-tokenizer",
        type=Path,
        help="Local FLUX CLIP tokenizer directory used for exact 77-token validation.",
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.verify_only:
            counts, clip_token_counts = verify_dataset(
                args.output, clip_tokenizer=args.clip_tokenizer
            )
        else:
            if args.source is None:
                parser.error("--source is required unless --verify-only is used")
            counts, clip_token_counts = build_dataset(
                args.source,
                args.output,
                clip_tokenizer=args.clip_tokenizer,
            )
    except ValueError as exc:
        parser.error(str(exc))
    print(
        f"DATASET_OK={EXPECTED_PAIRS} "
        f"TEMPLATE_COUNTS={json.dumps(counts, sort_keys=True)} "
        f"CLIP_TOKEN_COUNTS={json.dumps(clip_token_counts, sort_keys=True)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
