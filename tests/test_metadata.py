import json

import pytest
from pydantic import ValidationError

from face_destyle.data.metadata import read_jsonl, write_jsonl
from face_destyle.schemas import ImageRecord


def test_jsonl_round_trip_and_path_check(tmp_path):
    image = tmp_path / "image.png"
    image.touch()
    records = [ImageRecord(id="one", image_path=image, style_category="comic", source_id="one")]
    metadata = tmp_path / "metadata.jsonl"
    write_jsonl(records, metadata)
    loaded = read_jsonl(metadata, ImageRecord, check_paths=True, path_fields=("image_path",))
    assert loaded == records


def test_duplicate_id_is_rejected(tmp_path):
    metadata = tmp_path / "metadata.jsonl"
    row = {
        "id": "same",
        "image_path": "anything.png",
        "style_category": "ink",
        "source_id": "same",
    }
    metadata.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate id"):
        read_jsonl(metadata, ImageRecord)


def test_required_field_is_validated():
    with pytest.raises(ValidationError):
        ImageRecord.model_validate({"id": "x", "image_path": "x.png"})
