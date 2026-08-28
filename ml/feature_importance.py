from pathlib import Path

import pandas as pd
import joblib


ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    ROOT
    / "dataset"
    / "metadata"
    / "feature_dataset.csv"
)

MODEL_PATH = (
    ROOT
    / "ml"
    / "artifacts"
    / "random_forest.joblib"
)


EXCLUDED_COLUMNS = [
    "source_id",
    "split",
    "variant_id",
    "degradation",
    "severity",
    "severity_name",
    "is_degraded",
]


def main():

    df = pd.read_csv(
        DATASET_PATH
    )

    feature_columns = [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    model = joblib.load(
        MODEL_PATH
    )

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": (
                model.feature_importances_
            ),
        }
    )

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False,
        )
    )

    print()
    print(
        "FEATURE IMPORTANCE"
    )
    print(
        "=" * 60
    )

    print(
        importance
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()