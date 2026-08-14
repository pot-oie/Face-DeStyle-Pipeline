"""Global Canny ControlNet SDXL image-to-image backend for GPU execution."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from face_destyle.models import ModelAsset, ModelRegistry
from face_destyle.pipelines.diffusers_backend import (
    DiffusersBackend,
    DiffusersSettings,
    PipelineFactory,
)
from face_destyle.schemas import ImageRecord


@dataclass(frozen=True)
class CannyControlNetSettings(DiffusersSettings):
    """Validated runtime settings for the first global Canny comparison."""

    control_model: str = "canny_controlnet"
    controlnet_conditioning_scale: float = 0.8
    control_guidance_start: float = 0.0
    control_guidance_end: float = 1.0
    canny_low: int = 100
    canny_high: int = 200

    def validate(self) -> None:
        super().validate()
        if not self.control_model:
            raise ValueError("control_model must name a registered model")
        if not 0.0 < self.controlnet_conditioning_scale <= 2.0:
            raise ValueError("controlnet_conditioning_scale must be in (0, 2]")
        if not 0.0 <= self.control_guidance_start < self.control_guidance_end <= 1.0:
            raise ValueError("control guidance must satisfy 0 <= start < end <= 1")
        if not 0 <= self.canny_low < self.canny_high <= 255:
            raise ValueError("Canny thresholds must satisfy 0 <= low < high <= 255")


class CannyControlNetBackend(DiffusersBackend):
    """Apply global Canny conditioning while retaining the prompt-only img2img settings."""

    name = "canny"

    def __init__(
        self,
        settings: CannyControlNetSettings,
        styles_config: dict[str, Any],
        model_registry: ModelRegistry,
        *,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        super().__init__(settings, styles_config, model_registry, pipeline_factory=pipeline_factory)
        self.settings = settings
        self._resolved_control_asset: ModelAsset | None = None
        self._resolved_control_model_path: Path | None = None

    def _load_real_control_pipeline(
        self,
        model_path: str,
        control_model_path: str,
        load_kwargs: dict[str, Any],
    ) -> Any:
        try:
            import torch
            from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
        except ImportError as exc:
            raise RuntimeError(
                'Canny backend requires GPU dependencies: pip install -e ".[gpu,dev]"'
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; Canny ControlNet requires a GPU")
        dtype = getattr(torch, self.settings.dtype)
        controlnet = ControlNetModel.from_pretrained(
            control_model_path,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16",
            local_files_only=self.settings.local_files_only,
        )
        pipeline = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            model_path,
            controlnet=controlnet,
            torch_dtype=dtype,
            **load_kwargs,
        )
        pipeline.to(self.settings.device)
        if self.settings.enable_attention_slicing:
            pipeline.enable_attention_slicing()
        return pipeline

    def _create_pipeline(self, model_path: Path, load_kwargs: dict[str, Any]) -> Any:
        control_asset, control_path = self._resolve_registered_model(
            self.settings.control_model
        )
        self._resolved_control_asset = control_asset
        self._resolved_control_model_path = control_path
        factory = self._pipeline_factory or self._load_real_control_pipeline
        return factory(str(model_path), str(control_path), load_kwargs)

    def _thresholds_for(self, record: ImageRecord) -> tuple[int, int]:
        style = self.styles_config.get("styles", {}).get(record.style_category, {})
        low = int(style.get("canny_low", self.settings.canny_low))
        high = int(style.get("canny_high", self.settings.canny_high))
        if not 0 <= low < high <= 255:
            raise ValueError(
                f"invalid Canny thresholds for {record.style_category!r}: low={low}, high={high}"
            )
        return low, high

    def _preflight_additional_outputs(self, record: ImageRecord, output_dir: Path) -> None:
        control_path = output_dir / f"{record.id}.canny.png"
        if control_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing control image: {control_path}")

    def _inference_arguments(
        self,
        record: ImageRecord,
        initial_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        seed: int,
        output_dir: Path,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        arguments, metadata = super()._inference_arguments(
            record,
            initial_image,
            prompt,
            negative_prompt,
            seed,
            output_dir,
        )
        low, high = self._thresholds_for(record)
        rgb = np.asarray(initial_image)
        grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(grayscale, low, high)
        control_image = Image.fromarray(np.repeat(edges[:, :, None], 3, axis=2))
        control_path = output_dir / f"{record.id}.canny.png"
        control_image.save(control_path)
        arguments.update(
            {
                "control_image": control_image,
                "controlnet_conditioning_scale": self.settings.controlnet_conditioning_scale,
                "control_guidance_start": self.settings.control_guidance_start,
                "control_guidance_end": self.settings.control_guidance_end,
            }
        )
        metadata.update(
            {
                "canny_low": low,
                "canny_high": high,
                "control_image_path": str(control_path),
                "controlnet_conditioning_scale": self.settings.controlnet_conditioning_scale,
                "control_guidance_start": self.settings.control_guidance_start,
                "control_guidance_end": self.settings.control_guidance_end,
            }
        )
        return arguments, metadata

    def _baseline_metadata(self) -> dict[str, Any]:
        assert self._resolved_control_asset is not None
        assert self._resolved_control_model_path is not None
        return {
            "baseline": "global_canny_sdxl_controlnet_img2img",
            "structural_control": "canny",
            "control_model_asset": self._resolved_control_asset.name,
            "control_model_id": self._resolved_control_asset.model_id,
            "control_model_revision": self._resolved_control_asset.revision,
            "resolved_control_model_path": str(self._resolved_control_model_path),
        }
