"""Image loading helpers."""

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def image_array(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = load_rgb(path)
    if size is not None:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)
