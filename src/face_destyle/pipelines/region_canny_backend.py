"""Face-parsing-aware Canny conditioning for the SDXL ControlNet baseline."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from face_destyle.models import ModelAsset, ModelRegistry
from face_destyle.pipelines.canny_controlnet_backend import (
    CannyControlNetBackend,
    CannyControlNetSettings,
)
from face_destyle.pipelines.diffusers_backend import PipelineFactory
from face_destyle.schemas import ImageRecord

FaceParser = Callable[[Image.Image], np.ndarray]
FaceParserFactory = Callable[[str, str], FaceParser]

# CelebAMask-HQ labels that describe the head and recoverable facial geometry. Accessories,
# clothing, and background remain outside the full-strength region.
HEAD_LABELS = frozenset(range(1, 14)) | {17}


@dataclass(frozen=True)
class RegionCannySettings(CannyControlNetSettings):
    """Settings for one composite face/background Canny condition."""

    face_parsing_model: str = "face_parsing"
    background_edge_scale: float = 0.25
    face_mask_dilation: int = 9

    def validate(self) -> None:
        super().validate()
        if not self.face_parsing_model:
            raise ValueError("face_parsing_model must name a registered model")
        if not 0.0 <= self.background_edge_scale < 1.0:
            raise ValueError("background_edge_scale must be in [0, 1)")
        if self.face_mask_dilation < 0 or (
            self.face_mask_dilation != 0 and self.face_mask_dilation % 2 == 0
        ):
            raise ValueError("face_mask_dilation must be zero or a positive odd integer")


class RegionCannyBackend(CannyControlNetBackend):
    """Keep head-region edges strong while attenuating background edges."""

    name = "region_canny"

    def __init__(
        self,
        settings: RegionCannySettings,
        styles_config: dict[str, Any],
        model_registry: ModelRegistry,
        *,
        pipeline_factory: PipelineFactory | None = None,
        face_parser_factory: FaceParserFactory | None = None,
    ) -> None:
        super().__init__(settings, styles_config, model_registry, pipeline_factory=pipeline_factory)
        self.settings = settings
        self._face_parser_factory = face_parser_factory
        self._face_parser: FaceParser | None = None
        self._resolved_face_parsing_asset: ModelAsset | None = None
        self._resolved_face_parsing_path: Path | None = None
        self._face_parser_load_seconds: float | None = None

    def _load_real_face_parser(self, model_path: str, device: str) -> FaceParser:
        try:
            import torch
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        except ImportError as exc:
            raise RuntimeError(
                'Region Canny requires GPU dependencies: pip install -e ".[gpu,dev]"'
            ) from exc

        # This pinned model's older preprocessor metadata does not declare image_processor_type.
        # Use the architecture-specific loader so current Transformers does not need auto-detection.
        processor = SegformerImageProcessor.from_pretrained(model_path, local_files_only=True)
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_path,
            local_files_only=True,
        )
        model.to(device).eval()

        def parse(image: Image.Image) -> np.ndarray:
            inputs = processor(images=image, return_tensors="pt")
            inputs = {name: value.to(device) for name, value in inputs.items()}
            with torch.inference_mode():
                logits = model(**inputs).logits
                logits = torch.nn.functional.interpolate(
                    logits,
                    size=(image.height, image.width),
                    mode="bilinear",
                    align_corners=False,
                )
            return logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)

        return parse

    def _get_face_parser(self) -> FaceParser:
        if self._face_parser is None:
            asset, model_path = self._resolve_registered_model(self.settings.face_parsing_model)
            factory = self._face_parser_factory or self._load_real_face_parser
            started = time.perf_counter()
            self._face_parser = factory(str(model_path), self.settings.device)
            self._face_parser_load_seconds = time.perf_counter() - started
            self._resolved_face_parsing_asset = asset
            self._resolved_face_parsing_path = model_path
        return self._face_parser

    def _get_pipeline(self) -> Any:
        pipeline = super()._get_pipeline()
        self._get_face_parser()
        return pipeline

    def _preflight_additional_outputs(self, record: ImageRecord, output_dir: Path) -> None:
        super()._preflight_additional_outputs(record, output_dir)
        for suffix in ("face-mask", "region-canny"):
            path = output_dir / f"{record.id}.{suffix}.png"
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite region-control artifact: {path}")

    def _head_mask(self, labels: np.ndarray) -> np.ndarray:
        if labels.ndim != 2:
            raise ValueError(f"face parser must return a 2D label map, got shape {labels.shape}")
        mask = np.isin(labels, tuple(HEAD_LABELS)).astype(np.uint8) * 255
        if self.settings.face_mask_dilation:
            kernel = np.ones(
                (self.settings.face_mask_dilation, self.settings.face_mask_dilation),
                dtype=np.uint8,
            )
            mask = cv2.dilate(mask, kernel, iterations=1)
        fraction = float(np.count_nonzero(mask) / mask.size)
        if not 0.005 <= fraction <= 0.95:
            raise RuntimeError(f"implausible parsed head-mask fraction: {fraction:.4f}")
        return mask

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
        parser = self._get_face_parser()
        labels = parser(initial_image)
        mask = self._head_mask(labels)
        if mask.shape != (initial_image.height, initial_image.width):
            raise ValueError(
                "face parser label map must match the inference image: "
                f"got {mask.shape}, expected {(initial_image.height, initial_image.width)}"
            )
        mask_path = output_dir / f"{record.id}.face-mask.png"
        Image.fromarray(mask).save(mask_path)

        global_edges = np.asarray(arguments["control_image"].convert("L"), dtype=np.float32)
        weights = np.where(mask > 0, 1.0, self.settings.background_edge_scale)
        region_edges = np.rint(global_edges * weights).astype(np.uint8)
        region_image = Image.fromarray(np.repeat(region_edges[:, :, None], 3, axis=2))
        region_path = output_dir / f"{record.id}.region-canny.png"
        region_image.save(region_path)
        arguments["control_image"] = region_image
        global_canny_path = metadata.pop("control_image_path")

        assert self._resolved_face_parsing_asset is not None
        assert self._resolved_face_parsing_path is not None
        metadata.update(
            {
                "structural_control": "region_canny",
                "face_parsing_model_asset": self._resolved_face_parsing_asset.name,
                "face_parsing_model_id": self._resolved_face_parsing_asset.model_id,
                "face_parsing_revision": self._resolved_face_parsing_asset.revision,
                "resolved_face_parsing_path": str(self._resolved_face_parsing_path),
                "face_parser_load_seconds": self._face_parser_load_seconds,
                "face_mask_path": str(mask_path),
                "face_mask_fraction": float(np.count_nonzero(mask) / mask.size),
                "global_canny_image_path": global_canny_path,
                "control_image_path": str(region_path),
                "region_control_image_path": str(region_path),
                "background_edge_scale": self.settings.background_edge_scale,
                "face_mask_dilation": self.settings.face_mask_dilation,
                "face_label_ids": sorted(HEAD_LABELS),
            }
        )
        return arguments, metadata

    def _baseline_metadata(self) -> dict[str, Any]:
        metadata = super()._baseline_metadata()
        metadata.update(
            {
                "baseline": "region_canny_sdxl_controlnet_img2img",
                "structural_control": "region_canny",
            }
        )
        return metadata
