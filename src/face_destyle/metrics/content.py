"""Lazy local-only DINOv2 and CLIP pair metrics for AutoDL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


def _device_name(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return requested


class DinoV2PairMetric:
    name = "dinov2_cosine"

    def __init__(self, model_dir: str | Path, *, device: str = "auto") -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._torch = torch
        self.device = _device_name(torch, device)
        self.model_dir = Path(model_dir).resolve()
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_dir, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(self.model_dir, local_files_only=True)
        self.model.to(self.device).eval()

    def score(self, source: str | Path, generated: str | Path) -> float:
        images = [
            Image.open(source).convert("RGB"),
            Image.open(generated).convert("RGB"),
        ]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            output = self.model(**inputs)
            embeddings = output.last_hidden_state[:, 0]
            embeddings = self._torch.nn.functional.normalize(embeddings, dim=-1)
            return float((embeddings[0] * embeddings[1]).sum().item())

    def close(self) -> None:
        del self.model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


class ClipPairMetric:
    name = "clip_cosine"

    def __init__(self, model_dir: str | Path, *, device: str = "auto") -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.device = _device_name(torch, device)
        self.model_dir = Path(model_dir).resolve()
        self.processor = CLIPProcessor.from_pretrained(self.model_dir, local_files_only=True)
        self.model = CLIPModel.from_pretrained(self.model_dir, local_files_only=True)
        self.model.to(self.device).eval()

    def score(self, source: str | Path, generated: str | Path) -> float:
        images = [
            Image.open(source).convert("RGB"),
            Image.open(generated).convert("RGB"),
        ]
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with self._torch.inference_mode():
            embeddings = self.model.get_image_features(pixel_values=pixel_values)
            embeddings = self._torch.nn.functional.normalize(embeddings, dim=-1)
            return float((embeddings[0] * embeddings[1]).sum().item())

    def close(self) -> None:
        del self.model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
