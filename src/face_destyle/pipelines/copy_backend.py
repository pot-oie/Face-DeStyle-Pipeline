"""A no-op backend used only to validate pipeline wiring."""

import shutil
from pathlib import Path

from face_destyle.pipelines.base import DestylizationBackend
from face_destyle.schemas import DestylizationRecord, ImageRecord


class CopyBackend(DestylizationBackend):
    """Copy inputs byte-for-byte; this is not a research destylization method."""

    name = "copy"

    def run(self, record: ImageRecord, output_dir: Path, *, seed: int) -> DestylizationRecord:
        source = Path(record.image_path)
        if not source.exists():
            raise FileNotFoundError(source)
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{record.id}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        return DestylizationRecord(
            id=record.id,
            source_id=record.source_id,
            input_path=source,
            output_path=destination,
            style_category=record.style_category,
            backend=self.name,
            seed=seed,
            extra={"warning": "No-op copy backend for smoke testing only."},
        )
