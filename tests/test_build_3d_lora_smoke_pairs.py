import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_3d_lora_smoke_pairs.py"
SPEC = importlib.util.spec_from_file_location("build_3d_lora_smoke_pairs", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

DESTYLIZE_INSTRUCTION = MODULE.DESTYLIZE_INSTRUCTION
natural_prompt = MODULE.natural_prompt
parse_indices = MODULE.parse_indices
selected_indices = MODULE.selected_indices
write_metadata = MODULE.write_metadata


def test_selected_indices_rejects_out_of_range() -> None:
    assert list(selected_indices(1, 24)) == list(range(1, 25))
    with pytest.raises(ValueError, match="exceed"):
        selected_indices(20, 6)


def test_parse_indices_supports_sparse_preview_selection() -> None:
    assert parse_indices("1, 10,21") == (1, 10, 21)
    with pytest.raises(ValueError, match="unique"):
        parse_indices("1,1")
    with pytest.raises(ValueError, match="between"):
        parse_indices("25")


def test_natural_prompts_are_fictional_and_distinct() -> None:
    prompts = [natural_prompt(index) for index in range(1, 25)]
    assert len(set(prompts)) == 24
    assert all("fictional person" in prompt for prompt in prompts)


def test_write_metadata_uses_two_image_columns(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    for folder in (train_dir / "target", train_dir / "condition"):
        folder.mkdir(parents=True)
        (folder / "portrait-001.png").write_bytes(b"placeholder")

    destination = write_metadata(output_dir=tmp_path, indices=range(1, 2))
    row = json.loads(destination.read_text(encoding="utf-8"))

    assert row == {
        "target_file_name": "target/portrait-001.png",
        "condition_file_name": "condition/portrait-001.png",
        "instruction": DESTYLIZE_INSTRUCTION,
    }
