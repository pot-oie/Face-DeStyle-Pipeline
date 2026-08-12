"""Backend interface for destylization implementations."""

from abc import ABC, abstractmethod
from pathlib import Path

from face_destyle.schemas import DestylizationRecord, ImageRecord


class DestylizationBackend(ABC):
    name: str

    @abstractmethod
    def run(self, record: ImageRecord, output_dir: Path, *, seed: int) -> DestylizationRecord:
        """Destylize one record and return auditable output metadata."""
