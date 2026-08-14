import importlib.util
from pathlib import Path

from face_destyle.schemas import ImageRecord

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_flux_kontext_probe.py"
SPEC = importlib.util.spec_from_file_location("run_flux_kontext_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def record(source_id: str, style: str) -> ImageRecord:
    return ImageRecord(
        id=source_id,
        source_id=source_id,
        image_path=Path(f"/{source_id}.png"),
        style_category=style,
    )


def test_probe_selection_is_fixed_by_style_and_source_id():
    records = [
        record("comic-z", "comic"),
        record("watercolor-a", "watercolor"),
        record("ink-a", "ink"),
        record("3d-b", "3d_cartoon"),
        record("comic-a", "comic"),
        record("3d-a", "3d_cartoon"),
    ]

    assert [item.source_id for item in MODULE.select_probe_records(records, "all")] == [
        "3d-a",
        "comic-a",
        "ink-a",
        "watercolor-a",
    ]
    assert [item.source_id for item in MODULE.select_probe_records(records, "first")] == [
        "3d-a"
    ]
    assert [item.source_id for item in MODULE.select_probe_records(records, "remaining")] == [
        "comic-a",
        "ink-a",
        "watercolor-a",
    ]
