import csv
import importlib.util
from pathlib import Path

from PIL import Image

from face_destyle.data.pair_bank import load_pair_bank_source_list

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_pair_bank_review.py"


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
