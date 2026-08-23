"""Lightweight source lists for exploratory reconstruction pair banks."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from face_destyle.schemas import ImageRecord

PAIR_BANK_ROLES = {"candidate", "holdout", "rejected"}


@dataclass(frozen=True)
class PairBankSource:
    source_id: str
    image_path: Path
    style_category: str
    role: str
    notes: str = ""

    def as_image_record(self) -> ImageRecord:
        return ImageRecord(
            id=self.source_id,
            source_id=self.source_id,
            image_path=self.image_path,
            style_category=self.style_category,
        )


def load_pair_bank_source_list(path: Path, data_root: Path) -> list[PairBankSource]:
    """Load a human-readable CSV without imposing formal-manifest ceremony."""
    rows: list[PairBankSource] = []
    source_ids: set[str] = set()
    root = data_root.expanduser().resolve()
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, payload in enumerate(csv.DictReader(handle), start=2):
            source_id = (payload.get("source_id") or "").strip()
            raw_asset = (payload.get("asset_path") or "").strip()
            style_category = (payload.get("style_category") or "").strip()
            role = (payload.get("role") or "").strip()
            if (
                not source_id
                or not raw_asset
                or not style_category
                or role not in PAIR_BANK_ROLES
            ):
                raise ValueError(f"invalid source-list row at line {line_number}")
            if source_id in source_ids:
                raise ValueError(f"duplicate source_id in source list: {source_id}")
            relative = Path(raw_asset)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe asset_path for {source_id}: {relative}")
            asset = (root / relative).resolve()
            try:
                asset.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"asset_path escapes data root: {relative}") from exc
            if not asset.is_file():
                raise FileNotFoundError(f"missing source image: {asset}")
            source_ids.add(source_id)
            rows.append(
                PairBankSource(
                    source_id=source_id,
                    image_path=asset,
                    style_category=style_category,
                    role=role,
                    notes=(payload.get("notes") or "").strip(),
                )
            )
    if not rows:
        raise ValueError("source list is empty")
    return rows
