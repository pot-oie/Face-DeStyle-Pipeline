#!/usr/bin/env python3
"""Check lightweight or optional GPU dependencies without downloading anything."""

import argparse
import importlib.util
import os
import platform
import re


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="Also report optional GPU packages.")
    args = parser.parse_args()
    invalid_runtime_settings = []
    if "OMP_NUM_THREADS" in os.environ:
        omp_threads = os.environ["OMP_NUM_THREADS"]
        if re.fullmatch(r"[1-9][0-9]*", omp_threads) is None:
            invalid_runtime_settings.append(
                "OMP_NUM_THREADS must be a positive integer; unset the invalid value before GPU "
                "execution"
            )
    required = ["PIL", "cv2", "numpy", "pandas", "pydantic", "yaml", "tqdm"]
    optional = (
        [
            "torch",
            "diffusers",
            "transformers",
            "accelerate",
            "safetensors",
            "onnxruntime",
            "insightface",
        ]
        if args.gpu
        else []
    )
    print(f"Python: {platform.python_version()}")
    for package in required + optional:
        status = "available" if importlib.util.find_spec(package) else "missing"
        print(f"{package}: {status}")
    for message in invalid_runtime_settings:
        print(f"runtime setting: invalid ({message})")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    return 1 if missing or invalid_runtime_settings else 0


if __name__ == "__main__":
    raise SystemExit(main())
