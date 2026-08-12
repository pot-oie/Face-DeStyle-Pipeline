"""Planned DINO/CLIP content-preservation metric."""

from pathlib import Path


def content_similarity(source: str | Path, generated: str | Path) -> float:
    raise NotImplementedError(
        "DINO/CLIP content evaluation is planned for AutoDL and is not installed locally."
    )
