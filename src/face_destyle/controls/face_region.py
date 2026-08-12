"""Crude masks for smoke tests; not production face segmentation."""

from collections.abc import Sequence

import numpy as np


def bbox_mask(height: int, width: int, bbox: Sequence[int]) -> np.ndarray:
    """Return a manual (x1, y1, x2, y2) mask for wiring tests."""
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("bbox must lie within the image and have positive area")
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def center_face_mask(height: int, width: int, ratio: float = 0.6) -> np.ndarray:
    """Approximate a centered face region for smoke tests, not real face parsing."""
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1]")
    box_width, box_height = int(width * ratio), int(height * ratio)
    x1, y1 = (width - box_width) // 2, (height - box_height) // 2
    return bbox_mask(height, width, (x1, y1, x1 + box_width, y1 + box_height))
