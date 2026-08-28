from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    ROOT
    / "dataset"
    / "metadata"
    / "feature_dataset.csv"
)

ARTIFACT_DIR = (
    ROOT
    / "ml"
    / "artifacts"
)

MODEL_PATH = (
    ARTIFACT_DIR
    / "random_forest.joblib"
)

LABEL_ENCODER_PATH = (
    ARTIFACT_DIR
    / "label_encoder.joblib"
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


def load_dataset():

    df = pd.read_csv(
        DATASET_PATH
    )

    feature_columns = [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    return df, feature_columns


def main():

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df, feature_columns = (
        load_dataset()
    )

    train_df = df[
        df["split"] == "train"
    ].copy()

    val_df = df[
        df["split"] == "val"
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()

    print(
        "Dataset sizes:"
    )

    print(
        f"Train: {len(train_df)}"
    )

    print(
        f"Validation: {len(val_df)}"
    )

    print(
        f"Test: {len(test_df)}"
    )

    print()

    X_train = train_df[
        feature_columns
    ]

    X_val = val_df[
        feature_columns
    ]

    X_test = test_df[
        feature_columns
    ]

    y_train = train_df[
        "degradation"
    ]

    y_val = val_df[
        "degradation"
    ]

    y_test = test_df[
        "degradation"
    ]

    label_encoder = LabelEncoder()

    y_train_encoded = (
        label_encoder.fit_transform(
            y_train
        )
    )

    y_val_encoded = (
        label_encoder.transform(
            y_val
        )
    )

    y_test_encoded = (
        label_encoder.transform(
            y_test
        )
    )

    print(
        "Classes:"
    )

    for index, label in enumerate(
        label_encoder.classes_
    ):
        print(
            f"{index}: {label}"
        )

    print()

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print(
        "Training Random Forest..."
    )

    model.fit(
        X_train,
        y_train_encoded,
    )

    print(
        "Training complete."
    )

    val_predictions = model.predict(
        X_val
    )

    print()
    print(
        "Validation Results"
    )
    print(
        "=" * 60
    )

    print(
        classification_report(
            y_val_encoded,
            val_predictions,
            target_names=(
                label_encoder.classes_
            ),
            zero_division=0,
        )
    )

    test_predictions = model.predict(
        X_test
    )

    print()
    print(
        "Test Results"
    )
    print(
        "=" * 60
    )

    print(
        classification_report(
            y_test_encoded,
            test_predictions,
            target_names=(
                label_encoder.classes_
            ),
            zero_division=0,
        )
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    joblib.dump(
        label_encoder,
        LABEL_ENCODER_PATH,
    )

    print()
    print(
        "Artifacts saved:"
    )

    print(
        MODEL_PATH
    )

    print(
        LABEL_ENCODER_PATH
    )


if __name__ == "__main__":
    main()