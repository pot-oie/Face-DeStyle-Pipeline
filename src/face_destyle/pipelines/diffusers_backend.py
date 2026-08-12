"""Planned diffusers backend interface for AutoDL GPU execution."""

from pathlib import Path

from face_destyle.pipelines.base import DestylizationBackend
from face_destyle.schemas import DestylizationRecord, ImageRecord


class DiffusersBackend(DestylizationBackend):
    name = "diffusers"

    def run(self, record: ImageRecord, output_dir: Path, *, seed: int) -> DestylizationRecord:
        raise NotImplementedError(
            "Diffusers inference is reserved for AutoDL. Install .[gpu] and implement "
            "model-specific loading without placing weights in Git."
        )
