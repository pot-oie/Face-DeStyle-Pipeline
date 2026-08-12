"""Configurable OpenCV Canny structural control."""

import argparse
from pathlib import Path

import cv2


def extract_canny(image_path: str | Path, *, low: int = 100, high: int = 200):
    if not 0 <= low < high <= 255:
        raise ValueError("Canny thresholds must satisfy 0 <= low < high <= 255")
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return cv2.Canny(image, low, high)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract a Canny edge map from one image.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--low", type=int, default=100)
    parser.add_argument("--high", type=int, default=200)
    args = parser.parse_args(argv)
    edges = extract_canny(args.input, low=args.low, high=args.high)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), edges):
        raise OSError(f"Failed to save {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
