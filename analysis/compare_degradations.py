import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = ROOT / "dataset" / "metadata" / "feature_dataset.csv"

FEATURES = ["laplacian_variance", "mean_gradient", "noise_estimate", "mean_brightness", "dark_ratio", "bright_ratio", "black_clip_ratio", "white_clip_ratio", "contrast_range", "dynamic_range", "entropy", "texture_variation", "mean_saturation"]

def main():
    df = pd.read_csv(FEATURE_PATH)
    print("\nFEATURE MEANS BY DEGRADATION\n")
    print(df.groupby("degradation")[FEATURES].mean().round(3).to_string())
    print("\n\nFEATURE MEDIANS BY DEGRADATION\n")
    print(df.groupby("degradation")[FEATURES].median().round(3).to_string())

if __name__ == "__main__":
    main()
