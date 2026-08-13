"""Typed lazy metric contracts for AutoDL implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class PairMetric(ABC):
    name: str

    @abstractmethod
    def score(self, source: str | Path, generated: str | Path) -> float:
        """Return a calibrated score in [0, 1]."""


class StyleRemovalMetric(ABC):
    name: str

    @abstractmethod
    def score(self, generated: str | Path, style_category: str) -> float:
        """Return a calibrated style-removal score in [0, 1]."""
