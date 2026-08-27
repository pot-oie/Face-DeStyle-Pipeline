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
SHOWCASE_MANIFEST = (
    ROOT / "data/manifests/multistyle-routing/showcase_refinement_20.jsonl"
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


def test_showcase_refinement_manifest_is_balanced_subset_of_validation_bank():
    validation_records = read_jsonl(MANIFEST, DatasetManifestRecord)
    showcase_records = read_jsonl(SHOWCASE_MANIFEST, DatasetManifestRecord)

    assert len(showcase_records) == 20
    assert len({record.source_id for record in showcase_records}) == 20
    assert Counter(record.style_category for record in showcase_records) == {
        "3d_cartoon": 10,
        "needle_felt": 10,
    }
    assert {record.split for record in showcase_records} == {"extension"}
    assert {record.source_id for record in showcase_records} <= {
        record.source_id for record in validation_records
    }
