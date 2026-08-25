import importlib.util
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_origami_lora_v21_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_origami_lora_v21_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v21_builder_rewrites_only_instructions_and_balances_templates(tmp_path):
    source = tmp_path / "v2"
    train = source / "train"
    (train / "condition").mkdir(parents=True)
    (train / "target").mkdir()
    tag_sets = [
        ["hair_headwear", "clothing_bust"],
        ["scalp_neck", "bald"],
        ["clothing_bust", "compound"],
        ["elderly_wrinkles", "beard"],
        ["dark_skin", "strong_shadow"],
    ]
    rows = []
    for index in range(51):
        source_id = f"synthetic-{index:03d}"
        condition = train / "condition" / f"{source_id}.png"
        target = train / "target" / f"{source_id}.png"
        Image.new("RGB", (4, 4), (index, 0, 0)).save(condition)
        Image.new("RGB", (4, 4), (0, index, 0)).save(target)
        rows.append(
            {
                "target_file_name": f"target/{source_id}.png",
                "condition_file_name": f"condition/{source_id}.png",
                "instruction": f"old long instruction {index}",
                "source_id": source_id,
                "pair_source": "test",
                "difficulty_tags": tag_sets[index % len(tag_sets)],
            }
        )
    (train / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    output = tmp_path / "v21"
    counts, clip_counts = MODULE.build_dataset(source, output)
    rewritten = [
        json.loads(line)
        for line in (output / "train" / "metadata.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert counts == MODULE.EXPECTED_TEMPLATE_COUNTS
    assert clip_counts is None
    assert len({row["instruction"] for row in rewritten}) == 5
    assert all(
        len(row["instruction"].split()) <= MODULE.CONSERVATIVE_WORD_LIMIT
        for row in rewritten
    )
    assert {row["source_id"] for row in rewritten} == {
        row["source_id"] for row in rows
    }
    assert MODULE.verify_dataset(output) == (MODULE.EXPECTED_TEMPLATE_COUNTS, None)


def test_clip_token_validation_rejects_truncated_template():
    class FakeTokenizer:
        def __init__(self, overflow: bool):
            self.overflow = overflow

        def __call__(self, prompt, **_kwargs):
            count = 78 if self.overflow and prompt == MODULE.CORE_INSTRUCTION else 60
            return {"input_ids": list(range(count))}

    assert set(MODULE.validate_clip_token_lengths(FakeTokenizer(False))) == set(
        MODULE.TEMPLATES
    )
    try:
        MODULE.validate_clip_token_lengths(FakeTokenizer(True))
    except ValueError as exc:
        assert "exceed 77 tokens" in str(exc)
    else:
        raise AssertionError("a truncated CLIP template was accepted")
