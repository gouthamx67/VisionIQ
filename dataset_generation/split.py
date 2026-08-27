from __future__ import annotations

import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CLEAN_DIR = (
    ROOT
    / "dataset"
    / "raw"
    / "clean"
)

SPLIT_DIR = (
    ROOT
    / "dataset"
    / "splits"
)

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def collect_images() -> list[str]:

    images = sorted(
        str(path.relative_to(CLEAN_DIR))
        for path in CLEAN_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower()
        in SUPPORTED_EXTENSIONS
    )

    return images


def write_split(
    name: str,
    images: list[str],
) -> None:

    output = SPLIT_DIR / f"{name}.txt"

    with output.open("w") as file:

        for image in images:
            file.write(
                image + "\n"
            )


def create_splits() -> None:

    images = collect_images()

    if not images:
        raise RuntimeError(
            "No clean images found."
        )

    if abs(
        TRAIN_RATIO
        + VAL_RATIO
        + TEST_RATIO
        - 1.0
    ) > 1e-9:

        raise ValueError(
            "Split ratios must sum to 1."
        )

    rng = random.Random(SEED)

    rng.shuffle(images)

    total = len(images)

    train_end = int(
        total * TRAIN_RATIO
    )

    val_end = train_end + int(
        total * VAL_RATIO
    )

    train_images = images[:train_end]

    val_images = images[
        train_end:val_end
    ]

    test_images = images[
        val_end:
    ]

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_split(
        "train",
        train_images,
    )

    write_split(
        "val",
        val_images,
    )

    write_split(
        "test",
        test_images,
    )

    print("=" * 60)
    print("DATASET SPLIT COMPLETE")
    print("=" * 60)
    print(f"Total : {total}")
    print(f"Train : {len(train_images)}")
    print(f"Val   : {len(val_images)}")
    print(f"Test  : {len(test_images)}")
    print(f"Seed  : {SEED}")


if __name__ == "__main__":
    create_splits()