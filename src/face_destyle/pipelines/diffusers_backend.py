"""Prompt-only SDXL image-to-image baseline for AutoDL GPU execution."""

import os
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from face_destyle.filtering.prompt_rewriter import select_prompt
from face_destyle.pipelines.base import DestylizationBackend
from face_destyle.schemas import DestylizationRecord, ImageRecord

PipelineFactory = Callable[[str, dict[str, Any]], Any]


@dataclass(frozen=True)
class DiffusersSettings:
    """Validated runtime settings for the prompt-only SDXL baseline."""

    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    revision: str = "462165984030d82259a11f4367a4eed129e94a7b"
    device: str = "cuda"
    dtype: str = "bfloat16"
    height: int = 768
    width: int = 768
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    strength: float = 0.45
    batch_size: int = 1
    prompt_mode: str = "adaptive"
    enable_attention_slicing: bool = True

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "DiffusersSettings":
        known = {field.name for field in fields(cls)}
        settings = cls(**{key: value for key, value in values.items() if key in known})
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.device != "cuda":
            raise ValueError("The first Diffusers baseline currently requires device=cuda")
        if self.dtype not in {"float16", "bfloat16"}:
            raise ValueError("dtype must be float16 or bfloat16")
        if self.height % 8 or self.width % 8 or min(self.height, self.width) < 64:
            raise ValueError("height and width must be multiples of 8 and at least 64")
        if not 0.0 < self.strength <= 1.0:
            raise ValueError("strength must be in (0, 1]")
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        if self.batch_size != 1:
            raise ValueError("This baseline processes one image at a time; batch_size must be 1")
        if self.prompt_mode not in {"generic", "adaptive"}:
            raise ValueError("prompt_mode must be generic or adaptive")


class DiffusersBackend(DestylizationBackend):
    """Lazy-loading prompt-only img2img backend; no ControlNet or pose is applied."""

    name = "diffusers"

    def __init__(
        self,
        settings: DiffusersSettings,
        styles_config: dict[str, Any],
        *,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        settings.validate()
        self.settings = settings
        self.styles_config = styles_config
        self._uses_injected_pipeline = pipeline_factory is not None
        self._pipeline_factory = pipeline_factory or self._load_real_pipeline
        self._pipeline: Any | None = None

    @staticmethod
    def _hf_cache_dir() -> Path:
        hf_home = os.environ.get("HF_HOME")
        if not hf_home:
            raise RuntimeError(
                "HF_HOME must point to persistent server storage before loading model weights."
            )
        root = Path(hf_home).expanduser().resolve()
        cache = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", root / "hub")).expanduser().resolve()
        try:
            cache.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("HUGGINGFACE_HUB_CACHE must be located inside HF_HOME") from exc
        cache.mkdir(parents=True, exist_ok=True)
        return cache

    def _load_real_pipeline(self, model_id: str, load_kwargs: dict[str, Any]) -> Any:
        try:
            import torch
            from diffusers import AutoPipelineForImage2Image
        except ImportError as exc:
            raise RuntimeError(
                'Diffusers backend requires GPU dependencies: pip install -e ".[gpu,dev]"'
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; use backend=copy for local smoke tests")
        dtype = getattr(torch, self.settings.dtype)
        pipeline = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
            cache_dir=self._hf_cache_dir(),
            **load_kwargs,
        )
        pipeline.to(self.settings.device)
        if self.settings.enable_attention_slicing:
            pipeline.enable_attention_slicing()
        return pipeline

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            kwargs: dict[str, Any] = {
                "revision": self.settings.revision,
                "use_safetensors": True,
                "variant": "fp16",
            }
            self._pipeline = self._pipeline_factory(self.settings.model_id, kwargs)
        return self._pipeline

    def _generator(self, seed: int) -> Any:
        if self._uses_injected_pipeline:
            return seed
        try:
            import torch
        except ImportError:
            return seed
        return torch.Generator(device=self.settings.device).manual_seed(seed)

    def _prompt_for(self, record: ImageRecord) -> tuple[str, str]:
        adaptive = self.settings.prompt_mode == "adaptive"
        prompt = select_prompt(record.style_category, self.styles_config, adaptive=adaptive)
        style = self.styles_config.get("styles", {}).get(record.style_category, {})
        negative_prompt = str(style.get("negative_prompt", ""))
        return prompt, negative_prompt

    def run(self, record: ImageRecord, output_dir: Path, *, seed: int) -> DestylizationRecord:
        source = Path(record.image_path)
        if not source.exists():
            raise FileNotFoundError(source)
        prompt, negative_prompt = self._prompt_for(record)
        with Image.open(source) as image:
            initial_image = ImageOps.fit(
                ImageOps.exif_transpose(image).convert("RGB"),
                (self.settings.width, self.settings.height),
                method=Image.Resampling.LANCZOS,
            )
        result = self._get_pipeline()(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=initial_image,
            strength=self.settings.strength,
            num_inference_steps=self.settings.num_inference_steps,
            guidance_scale=self.settings.guidance_scale,
            generator=self._generator(seed),
            height=self.settings.height,
            width=self.settings.width,
        )
        if not getattr(result, "images", None):
            raise RuntimeError("Diffusers pipeline returned no images")
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{record.id}.png"
        result.images[0].save(destination)
        return DestylizationRecord(
            id=record.id,
            source_id=record.source_id,
            input_path=source,
            output_path=destination,
            style_category=record.style_category,
            backend=self.name,
            seed=seed,
            prompt=prompt,
            extra={
                "baseline": "prompt_only_sdxl_img2img",
                "model_id": self.settings.model_id,
                "revision": self.settings.revision,
                "negative_prompt": negative_prompt,
                "strength": self.settings.strength,
                "num_inference_steps": self.settings.num_inference_steps,
                "guidance_scale": self.settings.guidance_scale,
                "height": self.settings.height,
                "width": self.settings.width,
                "dtype": self.settings.dtype,
                "device": self.settings.device,
            },
        )
