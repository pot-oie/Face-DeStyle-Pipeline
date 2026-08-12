"""Placeholder for optional pose conditioning on AutoDL."""

from pathlib import Path

import numpy as np


def extract_pose_control(image_path: str | Path) -> np.ndarray:
    raise NotImplementedError(
        "Pose extraction requires a selected GPU-capable pose model and will be implemented "
        "on AutoDL."
    )
