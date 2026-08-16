"""Structured Qwen2.5-VL pair rubric with local-only loading."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

RUBRIC_VERSION = "face-destyle-paired-0to5-v1"


class QwenPairRubric:
    name = "qwen25_vl_pair_rubric"

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "auto",
        max_new_tokens: int = 160,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self._torch = torch
        self.max_new_tokens = max_new_tokens
        self.model_dir = Path(model_dir).resolve()
        device_map: str | None = "auto" if device == "auto" else None
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_dir,
            local_files_only=True,
            torch_dtype="auto",
            device_map=device_map,
        )
        if device != "auto":
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
            self.model.to(device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_dir, local_files_only=True
        )

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise ValueError("Qwen response did not contain a JSON object")
        payload = json.loads(match.group(0))
        for key in ("content_preservation", "style_removal", "identity_preservation"):
            value = payload.get(key)
            if isinstance(value, bool):
                raise ValueError(f"Qwen field {key} must be an integer in [0, 5]")
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            elif isinstance(value, str) and re.fullmatch(r"[0-5]", value.strip()):
                value = int(value.strip())
            if not isinstance(value, int) or not 0 <= value <= 5:
                raise ValueError(f"Qwen field {key} must be an integer in [0, 5]")
            payload[key] = value
        evidence = payload.get("evidence", "")
        if not isinstance(evidence, str):
            raise ValueError("Qwen field evidence must be a string")
        return payload

    def score(
        self,
        source: str | Path,
        generated: str | Path,
        style_category: str,
    ) -> tuple[dict[str, Any], str]:
        from qwen_vl_utils import process_vision_info

        instruction = (
            "You are rating a private face-destylization experiment. Image 1 is the stylized "
            f"source ({style_category}); image 2 is its generated reconstruction. Return JSON only "
            "with integer fields content_preservation, style_removal, identity_preservation from "
            "0 to 5, plus evidence containing at most 35 words of visible observations. Content 5 "
            "means pose/composition/objects preserved. Style removal 5 means photographic with no "
            "meaningful source-style residue. Identity 5 means visible identity-bearing geometry "
            "is consistent; this is not real-person identification."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{Path(source).resolve()}"},
                    {"type": "image", "image": f"file://{Path(generated).resolve()}"},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        target_device = next(self.model.parameters()).device
        inputs = inputs.to(target_device)
        with self._torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        raw = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return self._parse_json(raw), raw

    def close(self) -> None:
        del self.model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
