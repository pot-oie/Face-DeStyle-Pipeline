"""Original-BF16 FLUX.1 Kontext capability-probe backend."""

from __future__ import annotations

import hashlib
import importlib.metadata
import resource
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from face_destyle.filtering.prompt_rewriter import select_prompt
from face_destyle.pipelines.base import DestylizationBackend
from face_destyle.schemas import DestylizationRecord, ImageRecord

PipelineFactory = Callable[[Path], Any]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FluxKontextSettings:
    """Frozen settings for the native-resolution generator-capability probe."""

    model_dir: Path
    download_manifest: Path
    hash_manifest: Path
    source_revision: str = "master"
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 28
    guidance_scale: float = 2.5
    dtype: str = "bfloat16"
    batch_size: int = 1
    local_files_only: bool = True

    def validate(self) -> None:
        if not self.model_dir.is_dir():
            raise ValueError(f"model directory does not exist: {self.model_dir}")
        required = [
            "model_index.json",
            "scheduler",
            "text_encoder",
            "text_encoder_2",
            "tokenizer",
            "tokenizer_2",
            "transformer",
            "vae",
        ]
        missing = [item for item in required if not (self.model_dir / item).exists()]
        if missing:
            raise ValueError("incomplete Kontext model tree: " + ", ".join(missing))
        for manifest in (self.download_manifest, self.hash_manifest):
            if not manifest.is_file():
                raise ValueError(f"model acquisition manifest does not exist: {manifest}")
        if self.dtype != "bfloat16":
            raise ValueError("the initial Kontext probe requires original bfloat16 weights")
        if self.batch_size != 1:
            raise ValueError("the initial Kontext probe requires batch_size=1")
        if (self.height, self.width) != (1024, 1024):
            raise ValueError("the Kontext probe is frozen at native 1024x1024")
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        if not self.local_files_only:
            raise ValueError("Kontext must load from the verified local directory only")


class FluxKontextBackend(DestylizationBackend):
    """Native source-image-plus-instruction editing with BF16 model offload."""

    name = "flux1_kontext_dev_prompt_edit_bf16_offloaded"

    def __init__(
        self,
        settings: FluxKontextSettings,
        styles_config: dict[str, Any],
        *,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        settings.validate()
        self.settings = settings
        self.styles_config = styles_config
        self._pipeline_factory = pipeline_factory
        self._uses_injected_pipeline = pipeline_factory is not None
        self._pipeline: Any | None = None
        self._pipeline_load_seconds: float | None = None

    def _load_real_pipeline(self, model_dir: Path) -> Any:
        try:
            import torch
            from diffusers import FluxKontextPipeline
        except ImportError as exc:
            raise RuntimeError(
                'FLUX Kontext requires GPU dependencies: pip install -e ".[gpu,dev]"'
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; the Kontext probe must run on AutoDL GPU")
        pipeline = FluxKontextPipeline.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        pipeline.enable_model_cpu_offload()
        return pipeline

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            started = time.perf_counter()
            factory = self._pipeline_factory or self._load_real_pipeline
            self._pipeline = factory(self.settings.model_dir)
            self._pipeline_load_seconds = time.perf_counter() - started
        return self._pipeline

    @property
    def pipeline_loaded(self) -> bool:
        return self._pipeline is not None

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
    def _system_memory() -> dict[str, int | None]:
        values: dict[str, int] = {}
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                key, _, raw = line.partition(":")
                if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                    values[key] = int(raw.strip().split()[0]) * 1024
        return {
            "ram_total_bytes": values.get("MemTotal"),
            "ram_available_bytes": values.get("MemAvailable"),
            "swap_total_bytes": values.get("SwapTotal"),
            "swap_free_bytes": values.get("SwapFree"),
            "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }

    def _generator(self, seed: int) -> Any:
        if self._uses_injected_pipeline:
            return seed
        import torch

        return torch.Generator(device="cpu").manual_seed(seed)

    def run(self, record: ImageRecord, output_dir: Path, *, seed: int) -> DestylizationRecord:
        source = Path(record.image_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{record.id}.png"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
        prompt = select_prompt(record.style_category, self.styles_config, adaptive=True)
        with Image.open(source) as image:
            initial_image = ImageOps.fit(
                ImageOps.exif_transpose(image).convert("RGB"),
                (self.settings.width, self.settings.height),
                method=Image.Resampling.LANCZOS,
            )
        memory_before_load = self._system_memory()
        pipeline_was_loaded = self._pipeline is not None
        pipeline = self._get_pipeline()
        memory_before_inference = self._system_memory()
        torch_module = None
        gpu_metadata: dict[str, Any] = {}
        if not self._uses_injected_pipeline:
            import torch

            torch_module = torch
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = pipeline(
            image=initial_image,
            prompt=prompt,
            guidance_scale=self.settings.guidance_scale,
            num_inference_steps=self.settings.num_inference_steps,
            generator=self._generator(seed),
            height=self.settings.height,
            width=self.settings.width,
            max_area=self.settings.height * self.settings.width,
        )
        if torch_module is not None:
            torch_module.cuda.synchronize()
            gpu_metadata = {
                "gpu_name": torch_module.cuda.get_device_name(0),
                "cuda_version": torch_module.version.cuda,
                "peak_allocated_bytes": torch_module.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch_module.cuda.max_memory_reserved(),
            }
        inference_seconds = time.perf_counter() - started
        memory_after_inference = self._system_memory()
        if not getattr(result, "images", None):
            raise RuntimeError("FluxKontextPipeline returned no images")
        output_image = result.images[0]
        output_image.save(destination)
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
                "probe": "flux1_kontext_dev_generator_capability",
                "official_model_id": "black-forest-labs/FLUX.1-Kontext-dev",
                "transport_source": "modelscope_mirror",
                "transport_model_id": "black-forest-labs/FLUX.1-Kontext-dev",
                "source_revision": self.settings.source_revision,
                "resolved_model_path": str(self.settings.model_dir.resolve()),
                "download_manifest": str(self.settings.download_manifest.resolve()),
                "download_manifest_sha256": _file_sha256(self.settings.download_manifest),
                "hash_manifest": str(self.settings.hash_manifest.resolve()),
                "hash_manifest_sha256": _file_sha256(self.settings.hash_manifest),
                "dtype": self.settings.dtype,
                "batch_size": self.settings.batch_size,
                "height": self.settings.height,
                "width": self.settings.width,
                "max_area": self.settings.height * self.settings.width,
                "output_height": output_image.height,
                "output_width": output_image.width,
                "guidance_scale": self.settings.guidance_scale,
                "num_inference_steps": self.settings.num_inference_steps,
                "offload": "enable_model_cpu_offload",
                "local_files_only": self.settings.local_files_only,
                "pipeline_loaded_this_run": not pipeline_was_loaded,
                "pipeline_load_seconds": self._pipeline_load_seconds,
                "inference_seconds": inference_seconds,
                "package_versions": self._package_versions(),
                "memory_before_load": memory_before_load,
                "memory_before_inference": memory_before_inference,
                "memory_after_inference": memory_after_inference,
                **gpu_metadata,
            },
        )
