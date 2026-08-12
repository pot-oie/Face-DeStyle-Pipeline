"""Planned ArcFace identity-preservation metric."""

from pathlib import Path


def identity_similarity(source: str | Path, generated: str | Path) -> float:
    raise NotImplementedError(
        "ArcFace identity evaluation is planned for AutoDL and is not installed locally."
    )
