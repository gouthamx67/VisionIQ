from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    ROOT
    / "dataset"
    / "metadata"
    / "feature_dataset.csv"
)

OUTPUT = (
    ROOT
    / "analysis"
    / "plots"
    / "correlation_matrix.png"
)


def main():

    df = pd.read_csv(
        FEATURE_PATH
    )

    excluded = [
        "source_id",
        "split",
        "variant_id",
        "degradation",
        "severity_name",
    ]

    feature_df = df.drop(
        columns=excluded
    )

    correlation = (
        feature_df.corr(
            numeric_only=True
        )
    )

    plt.figure(
        figsize=(14, 12)
    )

    plt.imshow(
        correlation,
        aspect="auto",
    )

    plt.colorbar(
        label="Correlation"
    )

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=90,
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns,
    )

    plt.title(
        "Feature Correlation Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved {OUTPUT}"
    )


if __name__ == "__main__":
    main()