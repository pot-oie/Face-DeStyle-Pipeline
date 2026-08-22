"""Deterministic selection of generic or style-adaptive prompts."""

from typing import Any


def select_prompt(
    style_category: str, styles_config: dict[str, Any], *, adaptive: bool = True
) -> str:
    generic = (
        "Convert the portrait into a natural realistic photograph while preserving identity, "
        "pose, composition, and background structure."
    )
    if not adaptive:
        return generic
    styles = styles_config.get("styles", {})
    if style_category not in styles:
        raise KeyError(f"Unknown style category: {style_category}")
    return str(styles[style_category]["stage1_prompt"])


def select_stage_prompt(
    style_category: str, styles_config: dict[str, Any], *, stage: str
) -> str:
    """Select a declared stage prompt without silently falling back across stages."""
    if stage not in {"stage1", "stage2"}:
        raise ValueError(f"Unknown prompt stage: {stage}")
    styles = styles_config.get("styles", {})
    if style_category not in styles:
        raise KeyError(f"Unknown style category: {style_category}")
    key = f"{stage}_prompt"
    prompt = styles[style_category].get(key)
    if not prompt:
        raise KeyError(f"Style {style_category!r} has no {key}")
    return str(prompt)
