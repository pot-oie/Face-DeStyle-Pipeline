from collections import Counter
from pathlib import Path

from face_destyle.data.metadata import read_jsonl
from face_destyle.schemas import DatasetManifestRecord

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "data/manifests/multistyle-routing/non_origami_validation_137.jsonl"
)
GAP_MANIFEST = (
    ROOT / "data/manifests/multistyle-routing/missing_stage12_sources.jsonl"
)


def test_non_origami_validation_manifest_has_declared_style_counts():
    records = read_jsonl(MANIFEST, DatasetManifestRecord)

    assert len(records) == 137
    assert len({record.source_id for record in records}) == 137
    assert Counter(record.style_category for record in records) == {
        "comic": 24,
        "ink": 24,
        "watercolor": 24,
        "3d_cartoon": 24,
        "clay": 24,
        "needle_felt": 17,
    }
    assert {record.split for record in records} == {"extension"}
    assert all(record.style_category != "origami" for record in records)

    gap_records = read_jsonl(GAP_MANIFEST, DatasetManifestRecord)
    gap_2d_ids = {
        record.source_id
        for record in gap_records
        if record.style_category in {"comic", "ink", "watercolor"}
    }
    assert gap_2d_ids.isdisjoint({record.source_id for record in records})
