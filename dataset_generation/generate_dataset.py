from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2
import numpy as np

from degradations import (
    adjust_exposure,
    gaussian_blur,
    gaussian_noise,
    jpeg_compression,
    motion_blur,
)


ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "dataset" / "raw" / "clean"
SPLIT_DIR = ROOT / "dataset" / "splits"
OUTPUT_DIR = ROOT / "dataset" / "generated"
METADATA_DIR = ROOT / "dataset" / "metadata"

SEED = 42
DEGRADATION_TYPES = ["blur", "motion_blur", "underexposure", "overexposure", "noise", "compression"]
SEVERITY = 3  # Medium severity for all


def read_split(name: str) -> list[str]:
    path = SPLIT_DIR / f"{name}.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_image(relative_path: str) -> np.ndarray:
    path = CLEAN_DIR / relative_path
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def apply_degradation(image: np.ndarray, degradation: str, rng: np.random.Generator) -> np.ndarray:
    if degradation == "blur": return gaussian_blur(image, SEVERITY)
    if degradation == "motion_blur": return motion_blur(image, SEVERITY)
    if degradation == "underexposure": return adjust_exposure(image, SEVERITY, "underexposure")
    if degradation == "overexposure": return adjust_exposure(image, SEVERITY, "overexposure")
    if degradation == "noise": return gaussian_noise(image, SEVERITY, rng)
    if degradation == "compression": return jpeg_compression(image, SEVERITY)
    raise ValueError(f"Unknown degradation: {degradation}")


def generate_split(split: str, rng: np.random.Generator, writer) -> None:
    images = read_split(split)
    output_dir = OUTPUT_DIR / split
    output_dir.mkdir(parents=True, exist_ok=True)

    for relative_path in images:
        image = load_image(relative_path)
        source_id = Path(relative_path).stem

        # Clean
        clean_filename = f"{source_id}__clean.jpg"
        clean_path = output_dir / clean_filename
        cv2.imwrite(str(clean_path), image)
        writer.writerow({"source_id": source_id, "split": split, "variant_id": source_id + "__clean", "degradation": "none", "severity": 0, "severity_name": "none", "is_degraded": 0, "relative_path": str(clean_path.relative_to(ROOT))})

        # All Degradations
        for degradation in DEGRADATION_TYPES:
            degraded = apply_degradation(image, degradation, rng)
            variant_id = f"{source_id}__{degradation}__s{SEVERITY}"
            output_path = output_dir / f"{variant_id}.jpg"
            cv2.imwrite(str(output_path), degraded)
            writer.writerow({"source_id": source_id, "split": split, "variant_id": variant_id, "degradation": degradation, "severity": SEVERITY, "severity_name": "medium", "is_degraded": 1, "relative_path": str(output_path.relative_to(ROOT))})


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = METADATA_DIR / "degradation_metadata.csv"
    rng = np.random.default_rng(SEED)

    fieldnames = ["source_id", "split", "variant_id", "degradation", "severity", "severity_name", "is_degraded", "relative_path"]

    with metadata_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for split in ["train", "val", "test"]:
            generate_split(split, rng, writer)

    print(f"DEGRADATION DATASET GENERATED. Metadata: {metadata_path}")

if __name__ == "__main__":
    main()
