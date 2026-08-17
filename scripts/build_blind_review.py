#!/usr/bin/env python3
"""Build deterministic method-hidden paired contact sheets and scoring CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from face_destyle.data.manifests import load_dataset_manifest

PANEL_SIZE = 768
HEADER_HEIGHT = 44
GAP = 12


def parse_method(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("method must use METHOD=/path/to/images")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise ValueError("method must use METHOD=/path/to/images")
    return name.strip(), Path(raw_path).expanduser().resolve()


def render_pair(source: Path, candidate: Path, destination: Path) -> None:
    canvas = Image.new(
        "RGB",
        (PANEL_SIZE * 2 + GAP, PANEL_SIZE + HEADER_HEIGHT),
        color=(245, 245, 245),
    )
    for index, (path, label) in enumerate(
        ((source, "SOURCE"), (candidate, "CANDIDATE"))
    ):
        with Image.open(path) as image:
            panel = ImageOps.pad(
                ImageOps.exif_transpose(image).convert("RGB"),
                (PANEL_SIZE, PANEL_SIZE),
                method=Image.Resampling.LANCZOS,
                color=(232, 232, 232),
                centering=(0.5, 0.5),
            )
        left = index * (PANEL_SIZE + GAP)
        canvas.paste(panel, (left, HEADER_HEIGHT))
        ImageDraw.Draw(canvas).text((left + 12, 14), label, fill=(20, 20, 20))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="JPEG", quality=92, optimize=True)


def write_instructions(path: Path) -> None:
    path.write_text(
        """# Blinded face-destylization review

Method identity and source ID are hidden. Review SOURCE and CANDIDATE at the same displayed size.
Score each dimension from 0 to 5; do not infer a person's true appearance.

- Content: 5 preserves subject, pose, composition, objects, and spatial relations; 0 is unrelated.
- Style removal: 5 is predominantly photographic with no meaningful source-style residue; 0 is
  failed or unchanged artwork.
- Recoverable facial identity: 5 preserves visible identity-bearing geometry; 0 has no usable face.
- Set `identity_judgment_valid` to `no` when the source or output has insufficient facial evidence.
- `failure_types` is a semicolon-separated subset of: `structure_drift`, `identity_drift`,
  `artistic_contour_residual`, `material_render_residual`, `background_drift`, `no_usable_face`,
  `other`, or blank when no failure is observed.

Do not open `private/private_key.jsonl` until every score in both rounds has been frozen. Complete
round A before opening round B. The two rounds use different opaque IDs and order.
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("pilot", "calibration", "test", "extension"),
        default="pilot",
        help="Frozen manifest split represented by every method directory.",
    )
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        metavar="METHOD=IMAGE_DIR",
        help="Opaque-review method and directory containing SOURCE_ID.png; repeat per method.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"refusing to use non-empty output directory: {args.output_dir}")
    methods = [parse_method(value) for value in args.method]
    method_names = [name for name, _path in methods]
    if len(set(method_names)) != len(method_names):
        parser.error("method names must be unique")
    for name, directory in methods:
        if not directory.is_dir():
            parser.error(f"method image directory does not exist for {name}: {directory}")

    sources = load_dataset_manifest(
        args.manifest,
        data_root=args.data_root,
        split=args.split,
    )
    candidates: list[dict[str, str]] = []
    for source in sources:
        for method, directory in methods:
            candidate = directory / f"{source.source_id}.png"
            if not candidate.is_file():
                parser.error(f"missing candidate for {method}/{source.source_id}: {candidate}")
            candidates.append(
                {
                    "canonical_id": f"{method}:{source.source_id}",
                    "method": method,
                    "source_id": source.source_id,
                    "style_category": source.style_category,
                    "source_path": str(Path(source.image_path).resolve()),
                    "candidate_path": str(candidate),
                }
            )

    reviewer_root = args.output_dir / "reviewer"
    private_root = args.output_dir / "private"
    reviewer_root.mkdir(parents=True, exist_ok=True)
    private_root.mkdir(parents=True, exist_ok=True)
    write_instructions(reviewer_root / "INSTRUCTIONS.md")
    private_rows = []
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
    for round_index, round_name in enumerate(("a", "b")):
        ordered = list(candidates)
        random.Random(args.seed + round_index).shuffle(ordered)
        round_root = reviewer_root / f"round-{round_name}"
        rows = []
        for index, candidate in enumerate(ordered, start=1):
            blind_id = f"{round_name.upper()}-{index:04d}"
            relative_pair = Path("pairs") / f"{blind_id}.jpg"
            render_pair(
                Path(candidate["source_path"]),
                Path(candidate["candidate_path"]),
                round_root / relative_pair,
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
                    "pair_path": str(Path(f"round-{round_name}") / relative_pair),
                    **candidate,
                }
            )
        with (round_root / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    with (private_root / "private_key.jsonl").open("w", encoding="utf-8") as handle:
        for row in private_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (private_root / "build.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "source_count": len(sources),
                "method_count": len(methods),
                "candidate_count_per_round": len(candidates),
                "rounds": 2,
                "method_names": method_names,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Built two blinded rounds with {len(candidates)} pairs each in {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
