#!/usr/bin/env python3
"""Check lightweight or optional GPU dependencies without downloading anything."""

import argparse
import importlib.util
import platform


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="Also report optional GPU packages.")
    args = parser.parse_args()
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
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
