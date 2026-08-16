"""Lazy paired ArcFace drift diagnostic for the private AutoDL study."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class ArcFacePairMetric:
    name = "arcface_cosine"

    def __init__(self, model_dir: str | Path, *, provider: str = "auto") -> None:
        from insightface.app import FaceAnalysis

        model_path = Path(model_dir).resolve()
        if model_path.name != "buffalo_l" or model_path.parent.name != "models":
            raise ValueError("InsightFace path must end in models/buffalo_l")
        root = model_path.parent.parent
        if provider == "auto":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif provider == "cuda":
            providers = ["CUDAExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.app = FaceAnalysis(
            name=model_path.name,
            root=str(root),
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        self.app.prepare(ctx_id=0 if provider != "cpu" else -1, det_size=(640, 640))

    @staticmethod
    def _largest(faces: list[Any]) -> Any | None:
        if not faces:
            return None
        return max(
            faces,
            key=lambda face: float(
                (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
            ),
        )

    def score(self, source: str | Path, generated: str | Path) -> tuple[float | None, str]:
        source_image = cv2.imread(str(source))
        generated_image = cv2.imread(str(generated))
        if source_image is None or generated_image is None:
            raise ValueError("ArcFace could not decode one or both images")
        source_face = self._largest(self.app.get(source_image))
        generated_face = self._largest(self.app.get(generated_image))
        if source_face is None and generated_face is None:
            return None, "no_face_both"
        if source_face is None:
            return None, "no_face_source"
        if generated_face is None:
            return None, "no_face_generated"
        left = np.asarray(source_face.normed_embedding, dtype=np.float32)
        right = np.asarray(generated_face.normed_embedding, dtype=np.float32)
        return float(np.dot(left, right)), "ok_largest_face"
