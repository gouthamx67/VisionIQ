import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = ROOT / "dataset" / "metadata" / "feature_dataset.csv"

def main():
    df = pd.read_csv(FEATURE_PATH)
    feature_columns = [col for col in df.columns if col not in ["source_id", "split", "variant_id", "degradation", "severity_name"]]
    print("\nDATASET SHAPE")
    print(df.shape)
    print("\nDEGRADATION COUNTS")
    print(df["degradation"].value_counts())
    print("\nFEATURE SUMMARY")
    print(df[feature_columns].describe().T.to_string())

if __name__ == "__main__":
    main()
