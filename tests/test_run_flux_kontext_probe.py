import importlib.util
import json
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
    assert [item.source_id for item in MODULE.select_probe_records(records, "pilot")] == [
        "3d-a",
        "3d-b",
        "comic-a",
        "comic-z",
        "ink-a",
        "watercolor-a",
    ]


def test_resume_reads_completed_source_ids(tmp_path):
    records = tmp_path / "records.jsonl"
    records.write_text(
        "\n".join(
            [
                json.dumps({"source_id": "source-a"}),
                json.dumps({"source_id": "source-b"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert MODULE.completed_source_ids(records) == {"source-a", "source-b"}
    assert MODULE.completed_source_ids(tmp_path / "missing.jsonl") == set()
