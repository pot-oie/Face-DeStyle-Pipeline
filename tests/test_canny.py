import cv2
import numpy as np
import pytest

from face_destyle.controls.canny import extract_canny
from face_destyle.controls.face_region import center_face_mask


def test_canny_detects_programmatic_square(tmp_path):
    image = np.zeros((64, 64), dtype=np.uint8)
    cv2.rectangle(image, (16, 16), (48, 48), 255, thickness=-1)
    path = tmp_path / "square.png"
    assert cv2.imwrite(str(path), image)
    edges = extract_canny(path, low=50, high=150)
    assert edges.shape == image.shape
    assert np.count_nonzero(edges) > 0


def test_invalid_canny_thresholds_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="thresholds"):
        extract_canny(tmp_path / "none.png", low=200, high=100)


def test_center_mask_is_only_an_approximation():
    mask = center_face_mask(100, 80, ratio=0.5)
    assert mask.shape == (100, 80)
    assert 0 < np.count_nonzero(mask) < mask.size
