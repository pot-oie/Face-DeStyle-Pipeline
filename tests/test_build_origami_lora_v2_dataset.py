import csv
import importlib.util
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_origami_lora_v2_dataset.py"
MANIFEST = (
    ROOT / "data/manifests/multistyle-pair-bank/origami_hard_v2_selection.csv"
)
SPEC = importlib.util.spec_from_file_location("build_origami_lora_v2_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_tracked_selection_is_portable_and_freezes_28_accepts() -> None:
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8", newline="")))
    accepted = {row["source_id"] for row in rows if row["decision"] == "accept"}

    assert len(accepted) == 28
    assert not MODULE.REVIEW_EXCLUSIONS.intersection(accepted)
    assert {row["source_id"] for row in rows if row["decision"] != "accept"} == (
        MODULE.REVIEW_EXCLUSIONS
    )
    for row in rows:
        assert not Path(row["condition_path"]).is_absolute()
        assert not Path(row["target_path"]).is_absolute()
        assert ".." not in Path(row["condition_path"]).parts
        assert ".." not in Path(row["target_path"]).parts


def make_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    v1_root = tmp_path / "v1"
    v2_root = tmp_path / "v2"
    metadata = []
    v1_selection = []
    for index in range(1, 24):
        source_id = f"v1-{index:03d}"
        for role, color in (("condition", (200, 0, 0)), ("target", (0, 200, 0))):
            path = v1_root / "train" / role / f"{source_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (12, 12), color).save(path)
        metadata.append(
            {
                "source_id": source_id,
                "condition_file_name": f"condition/{source_id}.png",
                "target_file_name": f"target/{source_id}.png",
            }
        )
        v1_selection.append(
            {
                "source_id": source_id,
                "selected_target": "closed-teacher",
                "decision": "accept",
                "reviewer_notes": "Hair, beard, neck, and clothing were converted.",
            }
        )
    (v1_root / "train/metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metadata), encoding="utf-8"
    )
    write_csv(
        v1_root / "target_selection.csv",
        ("source_id", "selected_target", "decision", "reviewer_notes"),
        v1_selection,
    )

    manifest_rows = []
    accepted_ids = [f"v2-{index:03d}" for index in range(1, 29)]
    rejected_ids = sorted(MODULE.REVIEW_EXCLUSIONS)
    for source_id in accepted_ids + rejected_ids:
        for role, color in (("raw/sources", (0, 0, 200)), ("teacher/accepted", (200, 200, 0))):
            path = v2_root / role / f"{source_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (12, 12), color).save(path)
        manifest_rows.append(
            {
                "source_id": source_id,
                "condition_path": f"raw/sources/{source_id}.png",
                "target_path": f"teacher/accepted/{source_id}.png",
                "decision": (
                    "reject_composition_drift"
                    if source_id in MODULE.REVIEW_EXCLUSIONS
                    else "accept"
                ),
                "difficulty_tags": "hair_headwear,clothing_bust",
                "reviewer_notes": "fixture",
            }
        )
    selection = tmp_path / "selection.csv"
    write_csv(
        selection,
        (
            "source_id",
            "condition_path",
            "target_path",
            "decision",
            "difficulty_tags",
            "reviewer_notes",
        ),
        manifest_rows,
    )
    return v1_root, v2_root, selection


def test_builds_exact_51_with_correct_pair_direction_and_regional_instructions(tmp_path) -> None:
    v1_root, v2_root, selection = make_fixture(tmp_path)
    output = tmp_path / "output"

    count = MODULE.build_dataset(
        v1_root=v1_root,
        v2_root=v2_root,
        selection=selection,
        output_dir=output,
    )

    assert count == 51
    rows = [json.loads(line) for line in (output / "train/metadata.jsonl").read_text().splitlines()]
    assert len(rows) == 51
    assert len(list((output / "train/condition").glob("*.png"))) == 51
    assert len(list((output / "train/target").glob("*.png"))) == 51
    assert not MODULE.REVIEW_EXCLUSIONS.intersection(row["source_id"] for row in rows)
    assert all("complete folded-paper origami subject" in row["instruction"] for row in rows)
    assert all("Fully naturalize the hair" in row["instruction"] for row in rows)
    assert all("complete garment, shoulders, lower bust" in row["instruction"] for row in rows)
    with Image.open(output / "train/condition/v2-001.png") as condition:
        assert condition.getpixel((0, 0)) == (0, 0, 200)
    with Image.open(output / "train/target/v2-001.png") as target:
        assert target.getpixel((0, 0)) == (200, 200, 0)
    assert len(list((output / "contact-sheets").glob("page-*.jpg"))) == 5
    assert (output / "preview.jpg").is_file()


def test_rejects_absolute_manifest_paths(tmp_path) -> None:
    v1_root, v2_root, selection = make_fixture(tmp_path)
    rows = list(csv.DictReader(selection.open(encoding="utf-8", newline="")))
    rows[0]["condition_path"] = "/private/absolute/source.png"
    write_csv(selection, tuple(rows[0]), rows)

    try:
        MODULE.load_v2_selection(selection, v2_root)
    except ValueError as exc:
        assert "non-portable" in str(exc)
    else:
        raise AssertionError("absolute selection path was allowed")


def test_rejects_independent_exclusion_if_marked_accepted(tmp_path) -> None:
    _v1_root, v2_root, selection = make_fixture(tmp_path)
    rows = list(csv.DictReader(selection.open(encoding="utf-8", newline="")))
    for row in rows:
        if row["source_id"] == "origami-hard-v2-021":
            row["decision"] = "accept"
    write_csv(selection, tuple(rows[0]), rows)

    try:
        MODULE.load_v2_selection(selection, v2_root)
    except ValueError as exc:
        assert "review exclusions were accepted" in str(exc)
    else:
        raise AssertionError("independently rejected pair was allowed")
