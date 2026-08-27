from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "source"
)

CLEAN_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "clean"
)

MIN_WIDTH = 256
MIN_HEIGHT = 256

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def create_image_id(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()[:16]


def is_valid_image(path: Path) -> bool:
    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        return False

    height, width = image.shape[:2]

    if width < MIN_WIDTH:
        return False

    if height < MIN_HEIGHT:
        return False

    return True


def prepare_sources() -> None:
    SOURCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CLEAN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates = sorted(
        path
        for path in SOURCE_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower()
        in SUPPORTED_EXTENSIONS
    )

    print(f"Found {len(candidates)} candidate images.")

    accepted = 0
    rejected = 0

    for path in candidates:

        if not is_valid_image(path):
            rejected += 1
            print(f"REJECTED: {path.name}")
            continue

        image_id = create_image_id(path)

        destination = (
            CLEAN_DIR
            / f"{image_id}{path.suffix.lower()}"
        )

        if destination.exists():
            continue

        shutil.copy2(
            path,
            destination,
        )

        accepted += 1

    print()
    print("=" * 60)
    print("SOURCE PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Candidates : {len(candidates)}")
    print(f"Accepted   : {accepted}")
    print(f"Rejected   : {rejected}")


if __name__ == "__main__":
    prepare_sources()
