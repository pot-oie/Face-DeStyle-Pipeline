"""Transparent pixel similarity used only for end-to-end smoke testing."""

from pathlib import Path

import numpy as np

from face_destyle.utils.image import image_array, load_rgb


def smoke_test_similarity(first: str | Path, second: str | Path) -> float:
    reference = load_rgb(first)
    candidate = image_array(second, size=reference.size)
    reference_array = np.asarray(reference, dtype=np.float32)
    mean_absolute_error = np.abs(reference_array - candidate).mean()
    return float(np.clip(1.0 - mean_absolute_error / 255.0, 0.0, 1.0))
