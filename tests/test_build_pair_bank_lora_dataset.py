import csv
import importlib.util
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_pair_bank_lora_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_pair_bank_lora_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_builds_only_accepted_pairs_from_mixed_target_sources(tmp_path):
    data_root = tmp_path / "data"
    run_dir = tmp_path / "run"
    for source_id, color in (("source-a", "red"), ("source-b", "blue")):
        source = data_root / "raw" / f"{source_id}.png"
        source.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), color).save(source)
    (run_dir / "stage1/images").mkdir(parents=True)
    Image.new("RGB", (16, 16), "green").save(
        run_dir / "stage1/images/source-a.png"
    )
    (run_dir / "closed-teacher/images").mkdir(parents=True)
    Image.new("RGB", (16, 16), "yellow").save(
        run_dir / "closed-teacher/images/source-b.png"
    )
    source_list = tmp_path / "sources.csv"
    write_csv(
        source_list,
        ("source_id", "asset_path", "style_category", "role", "notes"),
        [
            {
                "source_id": source_id,
                "asset_path": f"raw/{source_id}.png",
                "style_category": "origami",
                "role": "candidate",
                "notes": "",
            }
            for source_id in ("source-a", "source-b")
        ],
    )
    selection = tmp_path / "selection.csv"
    write_csv(
        selection,
        ("source_id", "selected_target", "decision", "reviewer_notes"),
        [
            {
                "source_id": "source-a",
                "selected_target": "stage1",
                "decision": "accept",
                "reviewer_notes": "",
            },
            {
                "source_id": "source-b",
                "selected_target": "closed-teacher",
                "decision": "accept",
                "reviewer_notes": "",
            },
        ],
    )
    styles = tmp_path / "styles.yaml"
    styles.write_text(
        "styles:\n"
        "  origami:\n"
        "    stage1_prompt: naturalize paper portrait\n"
        "    stage2_prompt: refine\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"

    count = MODULE.build_dataset(
        source_list=source_list,
        selection=selection,
        data_root=data_root,
        run_dir=run_dir,
        output_dir=output,
        styles_config=styles,
        style_category="origami",
    )

    assert count == 2
    rows = [json.loads(line) for line in (output / "train/metadata.jsonl").read_text().splitlines()]
    assert [row["source_id"] for row in rows] == ["source-a", "source-b"]
    assert {row["instruction"] for row in rows} == {"naturalize paper portrait"}
    assert (output / "train/condition/source-a.png").is_file()
    assert (output / "train/target/source-b.png").is_file()
    assert (output / "preview.jpg").is_file()


def test_rejects_duplicate_accepted_source_ids(tmp_path):
    selection = tmp_path / "selection.csv"
    write_csv(
        selection,
        ("source_id", "selected_target", "decision"),
        [
            {"source_id": "source-a", "selected_target": "stage1", "decision": "accept"},
            {"source_id": "source-a", "selected_target": "stage1", "decision": "accept"},
        ],
    )

    try:
        MODULE.load_accepted_selection(selection)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate accepted source ID was allowed")
