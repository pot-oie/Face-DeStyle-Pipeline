import json
from collections import Counter
from pathlib import Path

from face_destyle.schemas import DatasetManifestRecord

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "manifests" / "formal-v1"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_formal_v1_manifest_is_balanced_and_split_disjoint() -> None:
    rows = read_jsonl(MANIFEST_DIR / "inputs.jsonl")
    records = [DatasetManifestRecord.model_validate(row) for row in rows]

    assert len(records) == 120
    assert Counter(record.split for record in records) == {
        "pilot": 20,
        "calibration": 40,
        "test": 60,
    }
    assert Counter((record.split, record.style_category) for record in records) == {
        ("pilot", "3d_cartoon"): 5,
        ("pilot", "comic"): 5,
        ("pilot", "ink"): 5,
        ("pilot", "watercolor"): 5,
        ("calibration", "3d_cartoon"): 10,
        ("calibration", "comic"): 10,
        ("calibration", "ink"): 10,
        ("calibration", "watercolor"): 10,
        ("test", "3d_cartoon"): 15,
        ("test", "comic"): 15,
        ("test", "ink"): 15,
        ("test", "watercolor"): 15,
    }

    source_splits: dict[str, str] = {}
    group_splits: dict[str, str] = {}
    checksum_splits: dict[str, str] = {}
    for record in records:
        assert source_splits.setdefault(record.source_id, record.split) == record.split
        assert group_splits.setdefault(record.source_group_id, record.split) == record.split
        assert checksum_splits.setdefault(record.sha256, record.split) == record.split
    assert len(source_splits) == 120
    assert len(group_splits) == 120
    assert len(checksum_splits) == 120


def test_formal_v1_public_provenance_matches_manifest_without_local_paths() -> None:
    inputs = {row["source_id"]: row for row in read_jsonl(MANIFEST_DIR / "inputs.jsonl")}
    provenance = {
        row["source_id"]: row
        for row in read_jsonl(MANIFEST_DIR / "provenance.public.jsonl")
    }

    assert provenance.keys() == inputs.keys()
    for source_id, row in provenance.items():
        assert row["sha256"] == inputs[source_id]["sha256"]
        assert row["source_group_id"] == inputs[source_id]["source_group_id"]
        assert row["split"] == inputs[source_id]["split"]
        assert "local_path" not in row
        assert "image_url" not in row
        assert "generation" not in row
        assert "phash" not in row
