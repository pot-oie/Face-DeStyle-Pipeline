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
