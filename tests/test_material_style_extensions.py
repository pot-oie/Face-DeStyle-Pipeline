from pathlib import Path

from face_destyle.filtering.prompt_rewriter import select_prompt
from face_destyle.utils.io import load_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_material_style_extensions_have_complete_prompt_configuration() -> None:
    styles = load_yaml(ROOT / "configs" / "styles.yaml")

    for name in ("3d_cartoon", "clay", "needle_felt"):
        config = styles["styles"][name]
        assert config["display_name"]
        assert config["stage1_prompt"] == select_prompt(name, styles, adaptive=True)
        assert config["stage2_prompt"]
        assert config["negative_prompt"]
        assert 0 <= config["canny_low"] < config["canny_high"] <= 255
