"""Prompt-only SDXL image-to-image baseline for AutoDL GPU execution."""

import importlib.metadata
import time
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from face_destyle.filtering.prompt_rewriter import select_prompt
from face_destyle.models import ModelAsset, ModelRegistry
from face_destyle.pipelines.base import DestylizationBackend
from face_destyle.schemas import DestylizationRecord, ImageRecord

PipelineFactory = Callable[..., Any]


@dataclass(frozen=True)
class DiffusersSettings:
    """Validated runtime settings for the prompt-only SDXL baseline."""

    model_asset: str = "sdxl_base"
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
    local_files_only: bool = True

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "DiffusersSettings":
        known = {field.name for field in fields(cls)}
        settings = cls(**{key: value for key, value in values.items() if key in known})
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.model_asset:
            raise ValueError("model_asset must name a registered model")
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
        if not self.local_files_only:
            raise ValueError("real model loading must keep local_files_only=true")


class DiffusersBackend(DestylizationBackend):
    """Lazy-loading prompt-only img2img backend; no ControlNet or pose is applied."""

    name = "diffusers"

    def __init__(
        self,
        settings: DiffusersSettings,
        styles_config: dict[str, Any],
        model_registry: ModelRegistry,
        *,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        settings.validate()
        self.settings = settings
        self.styles_config = styles_config
        self.model_registry = model_registry
        self._uses_injected_pipeline = pipeline_factory is not None
        self._pipeline_factory = pipeline_factory
        self._pipeline: Any | None = None
        self._resolved_asset: ModelAsset | None = None
        self._resolved_model_path: Path | None = None
        self._pipeline_load_seconds: float | None = None

    def _load_real_pipeline(self, model_path: str, load_kwargs: dict[str, Any]) -> Any:
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
            model_path,
            torch_dtype=dtype,
            **load_kwargs,
        )
        pipeline.to(self.settings.device)
        if self.settings.enable_attention_slicing:
            pipeline.enable_attention_slicing()
        return pipeline

    def _resolve_model(self) -> tuple[ModelAsset, Path]:
        return self._resolve_registered_model(self.settings.model_asset)

    def _resolve_registered_model(self, name: str) -> tuple[ModelAsset, Path]:
        asset = self.model_registry.require(name)
        if asset.loader != "from_pretrained":
            raise RuntimeError(
                f"model asset {asset.name} requires unsupported loader {asset.loader!r}"
            )
        check = self.model_registry.check(asset.name)
        if not check.available or check.location is None:
            details = check.reason or "required files are unavailable"
            if check.missing_files:
                details = "missing files: " + ", ".join(check.missing_files)
            raise RuntimeError(f"model asset {asset.name} is unavailable: {details}")
        return asset, check.location

    def _create_pipeline(self, model_path: Path, load_kwargs: dict[str, Any]) -> Any:
        factory = self._pipeline_factory or self._load_real_pipeline
        return factory(str(model_path), load_kwargs)

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            asset, model_path = self._resolve_model()
            kwargs: dict[str, Any] = {
                "use_safetensors": True,
                "variant": "fp16",
                "local_files_only": self.settings.local_files_only,
            }
            started = time.perf_counter()
            self._pipeline = self._create_pipeline(model_path, kwargs)
            self._pipeline_load_seconds = time.perf_counter() - started
            self._resolved_asset = asset
            self._resolved_model_path = model_path
        return self._pipeline

    @staticmethod
    def _package_versions() -> dict[str, str]:
        versions = {}
        for package in (
            "face-destyle-pipeline",
            "torch",
            "diffusers",
            "transformers",
            "accelerate",
            "huggingface-hub",
        ):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "not-installed"
        return versions

    @staticmethod
    def _scheduler_metadata(pipeline: Any) -> dict[str, Any]:
        scheduler = getattr(pipeline, "scheduler", None)
        if scheduler is None:
            return {"class": "unavailable", "config": {}}
        config = getattr(scheduler, "config", {})
        selected = {}
        for key in (
            "beta_start",
            "beta_end",
            "beta_schedule",
            "prediction_type",
            "timestep_spacing",
            "steps_offset",
        ):
            value = config.get(key) if hasattr(config, "get") else None
            if isinstance(value, str | int | float | bool) or value is None:
                selected[key] = value
        return {"class": type(scheduler).__name__, "config": selected}

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

    def _inference_arguments(
        self,
        record: ImageRecord,
        initial_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        seed: int,
        output_dir: Path,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del record, output_dir
        return (
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": initial_image,
                "strength": self.settings.strength,
                "num_inference_steps": self.settings.num_inference_steps,
                "guidance_scale": self.settings.guidance_scale,
                "generator": self._generator(seed),
                "height": self.settings.height,
                "width": self.settings.width,
            },
            {},
        )

    def _baseline_metadata(self) -> dict[str, Any]:
        return {"baseline": "prompt_only_sdxl_img2img"}

    def _preflight_additional_outputs(self, record: ImageRecord, output_dir: Path) -> None:
        del record, output_dir

    def run(self, record: ImageRecord, output_dir: Path, *, seed: int) -> DestylizationRecord:
        source = Path(record.image_path)
        if not source.exists():
            raise FileNotFoundError(source)
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{record.id}.png"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
        self._preflight_additional_outputs(record, output_dir)
        prompt, negative_prompt = self._prompt_for(record)
        with Image.open(source) as image:
            initial_image = ImageOps.fit(
                ImageOps.exif_transpose(image).convert("RGB"),
                (self.settings.width, self.settings.height),
                method=Image.Resampling.LANCZOS,
            )
        pipeline_was_loaded = self._pipeline is not None
        pipeline = self._get_pipeline()
        torch_module = None
        gpu_metadata: dict[str, Any] = {}
        if not self._uses_injected_pipeline:
            import torch

            torch_module = torch
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        inference_arguments, inference_metadata = self._inference_arguments(
            record,
            initial_image,
            prompt,
            negative_prompt,
            seed,
            output_dir,
        )
        result = pipeline(**inference_arguments)
        if torch_module is not None:
            torch_module.cuda.synchronize()
            gpu_metadata = {
                "gpu_name": torch_module.cuda.get_device_name(0),
                "cuda_version": torch_module.version.cuda,
                "peak_allocated_bytes": torch_module.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch_module.cuda.max_memory_reserved(),
            }
        inference_seconds = time.perf_counter() - started
        if not getattr(result, "images", None):
            raise RuntimeError("Diffusers pipeline returned no images")
        result.images[0].save(destination)
        assert self._resolved_asset is not None
        assert self._resolved_model_path is not None
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
                **self._baseline_metadata(),
                "model_asset": self._resolved_asset.name,
                "model_id": self._resolved_asset.model_id,
                "revision": self._resolved_asset.revision,
                "resolved_model_path": str(self._resolved_model_path),
                "negative_prompt": negative_prompt,
                "prompt_mode": self.settings.prompt_mode,
                "strength": self.settings.strength,
                "num_inference_steps": self.settings.num_inference_steps,
                "guidance_scale": self.settings.guidance_scale,
                "height": self.settings.height,
                "width": self.settings.width,
                "dtype": self.settings.dtype,
                "device": self.settings.device,
                "local_files_only": self.settings.local_files_only,
                "pipeline_loaded_this_run": not pipeline_was_loaded,
                "pipeline_load_seconds": self._pipeline_load_seconds,
                "inference_seconds": inference_seconds,
                "scheduler": self._scheduler_metadata(pipeline),
                "package_versions": self._package_versions(),
                **inference_metadata,
                **gpu_metadata,
            },
        )
