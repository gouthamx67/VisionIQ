from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


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

LABEL_ENCODER_PATH = (
    ROOT
    / "ml"
    / "artifacts"
    / "label_encoder.joblib"
)

PLOT_PATH = (
    ROOT
    / "analysis"
    / "plots"
    / "confusion_matrix.png"
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

    test_df = df[
        df["split"] == "test"
    ].copy()

    feature_columns = [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    X_test = test_df[
        feature_columns
    ]

    y_test = test_df[
        "degradation"
    ]

    model = joblib.load(
        MODEL_PATH
    )

    label_encoder = joblib.load(
        LABEL_ENCODER_PATH
    )

    y_test_encoded = (
        label_encoder.transform(
            y_test
        )
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test_encoded,
        predictions,
    )

    print()
    print(
        f"Test accuracy: {accuracy:.4f}"
    )

    print()
    print(
        classification_report(
            y_test_encoded,
            predictions,
            target_names=(
                label_encoder.classes_
            ),
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_test_encoded,
        predictions,
    )

    print(
        "Confusion Matrix:"
    )

    print(matrix)

    display = (
        ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=(
                label_encoder.classes_
            ),
        )
    )

    display.plot(
        xticks_rotation=45
    )

    plt.title(
        "Random Forest Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_PATH,
        dpi=150,
    )

    plt.close()

    print()
    print(
        f"Saved: {PLOT_PATH}"
    )


if __name__ == "__main__":
    main()