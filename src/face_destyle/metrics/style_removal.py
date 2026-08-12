"""Planned VLM-based style-removal metric."""

from pathlib import Path


def style_removal_score(generated: str | Path, style_category: str) -> float:
    raise NotImplementedError(
        "VLM style-removal evaluation is planned for AutoDL and is not installed locally."
    )
