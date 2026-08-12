from PIL import Image

from face_destyle.pipelines.copy_backend import CopyBackend
from face_destyle.schemas import ImageRecord


def test_copy_backend_copies_bytes_and_marks_backend(tmp_path):
    source = tmp_path / "input.png"
    Image.new("RGB", (16, 16), (20, 40, 60)).save(source)
    record = ImageRecord(id="sample", source_id="sample", image_path=source, style_category="ink")
    result = CopyBackend().run(record, tmp_path / "out", seed=42)
    assert result.backend == "copy"
    assert result.seed == 42
    assert result.output_path.read_bytes() == source.read_bytes()
    assert "smoke testing" in result.extra["warning"]
