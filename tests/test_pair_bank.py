import csv
import importlib.util
from collections import Counter
from pathlib import Path

from PIL import Image

from face_destyle.data.pair_bank import load_pair_bank_source_list

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_pair_bank_review.py"
SOURCE_LIST_ROOT = ROOT / "data/manifests/multistyle-pair-bank"


def write_source_list(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("source_id", "asset_path", "style_category", "role", "notes"),
        )
        writer.writeheader()
        writer.writerows(rows)


def test_pair_bank_source_list_loads_lightweight_roles(tmp_path):
    data_root = tmp_path / "data"
    image = data_root / "raw" / "one.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (24, 24), "red").save(image)
    source_list = tmp_path / "sources.csv"
    write_source_list(
        source_list,
        [
            {
                "source_id": "one",
                "asset_path": "raw/one.png",
                "style_category": "3d_cartoon",
                "role": "candidate",
                "notes": "clear portrait",
            }
        ],
    )

    rows = load_pair_bank_source_list(source_list, data_root)

    assert rows[0].image_path == image.resolve()
    assert rows[0].role == "candidate"
    assert rows[0].as_image_record().style_category == "3d_cartoon"


def test_pair_bank_role_filter_does_not_require_rejected_assets(tmp_path):
    data_root = tmp_path / "data"
    candidate = data_root / "raw" / "candidate.png"
    candidate.parent.mkdir(parents=True)
    Image.new("RGB", (16, 16), "green").save(candidate)
    source_list = tmp_path / "sources.csv"
    write_source_list(
        source_list,
        [
            {
                "source_id": "candidate",
                "asset_path": "raw/candidate.png",
                "style_category": "3d_cartoon",
                "role": "candidate",
                "notes": "",
            },
            {
                "source_id": "rejected",
                "asset_path": "raw/not-transferred.png",
                "style_category": "3d_cartoon",
                "role": "rejected",
                "notes": "inventory only",
            },
        ],
    )

    rows = load_pair_bank_source_list(source_list, data_root, roles={"candidate"})

    assert [row.source_id for row in rows] == ["candidate"]


def test_pair_bank_review_builds_layout_and_four_column_sheet(tmp_path):
    data_root = tmp_path / "data"
    source = data_root / "raw" / "one.png"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (32, 48), "red").save(source)
    source_list = tmp_path / "sources.csv"
    write_source_list(
        source_list,
        [
            {
                "source_id": "one",
                "asset_path": "raw/one.png",
                "style_category": "3d_cartoon",
                "role": "candidate",
                "notes": "",
            }
        ],
    )
    run_dir = tmp_path / "run"

    spec = importlib.util.spec_from_file_location("build_pair_bank_review", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.load_pair_bank_source_list(source_list, data_root)
    module.prepare_layout(run_dir, rows)
    Image.new("RGB", (48, 32), "blue").save(run_dir / "stage1/images/one.png")
    page_count = module.build_review_pages(rows, run_dir, rows_per_page=6)
    module.build_contact_sheet(rows, run_dir / "review/inventory.jpg")

    assert page_count == 1
    sheet = run_dir / "review/sheets/review-01.jpg"
    assert sheet.is_file()
    with Image.open(sheet) as image:
        assert image.width == module.REVIEW_SIZE * 4 + module.GAP * 3
    assert (run_dir / "stage2-sequential/images").is_dir()
    assert (run_dir / "closed-teacher/NOTES.md").is_file()
    assert (run_dir / "review/target_selection.csv").is_file()


def test_curated_pair_bank_lists_have_declared_role_counts():
    expected = {
        "3d_cartoon_sources.csv": ("3d_cartoon", Counter(candidate=27, holdout=6, rejected=34)),
        "clay_sources.csv": ("clay", Counter(candidate=19, holdout=5, rejected=12)),
        "origami_sources.csv": ("origami", Counter(candidate=24, holdout=6)),
    }
    for filename, (style, role_counts) in expected.items():
        with (SOURCE_LIST_ROOT / filename).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert {row["style_category"] for row in rows} == {style}
        assert Counter(row["role"] for row in rows) == role_counts
        assert len({row["source_id"] for row in rows}) == len(rows)
