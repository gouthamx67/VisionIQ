from __future__ import annotations

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
from pathlib import Path

import cv2
import pandas as pd

from vision.features import extract_features


ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = (
    ROOT
    / "dataset"
    / "metadata"
    / "degradation_metadata.csv"
)

OUTPUT_PATH = (
    ROOT
    / "dataset"
    / "metadata"
    / "feature_dataset.csv"
)


def load_metadata() -> pd.DataFrame:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}"
        )
    return pd.read_csv(METADATA_PATH)


def extract_dataset_features(
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    total = len(metadata)

    for index, row in metadata.iterrows():
        image_path = ROOT / row["relative_path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            print(f"WARNING: Could not read {image_path}")
            continue

        features = extract_features(image)

        result = {
            "source_id": row["source_id"],
            "split": row["split"],
            "variant_id": row["variant_id"],
            "degradation": row["degradation"],
            "severity": row["severity"],
            "severity_name": row["severity_name"],
            "is_degraded": row["is_degraded"],
        }

        result.update(features)
        rows.append(result)

        if (index + 1) % 25 == 0:
            print(f"Processed {index + 1}/{total}")

    return pd.DataFrame(rows)


def main():
    metadata = load_metadata()
    print(f"Found {len(metadata)} examples.")

    feature_df = extract_dataset_features(metadata)
    feature_df.to_csv(OUTPUT_PATH, index=False)

    print()
    print("=" * 60)
    print("FEATURE DATASET CREATED")
    print("=" * 60)
    print(f"Rows: {len(feature_df)}")
    print(f"Columns: {len(feature_df.columns)}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
