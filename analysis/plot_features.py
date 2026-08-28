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

PLOT_DIR = (
    ROOT
    / "analysis"
    / "plots"
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def plot_feature(
    df: pd.DataFrame,
    feature: str,
):

    plt.figure(
        figsize=(10, 6)
    )

    for degradation in sorted(
        df["degradation"]
        .unique()
    ):

        values = df.loc[
            df["degradation"]
            == degradation,
            feature,
        ]

        plt.hist(
            values,
            bins=30,
            alpha=0.5,
            label=degradation,
        )

    plt.title(
        f"Distribution of {feature}"
    )

    plt.xlabel(feature)

    plt.ylabel(
        "Number of images"
    )

    plt.legend()

    plt.tight_layout()

    output = (
        PLOT_DIR
        / f"{feature}.png"
    )

    plt.savefig(
        output,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved {output}"
    )


def main():

    df = pd.read_csv(
        FEATURE_PATH
    )

    features = [
        "laplacian_variance",
        "mean_gradient",
        "noise_estimate",
        "mean_brightness",
        "dark_ratio",
        "bright_ratio",
        "entropy",
    ]

    for feature in features:

        plot_feature(
            df,
            feature,
        )


if __name__ == "__main__":
    main()