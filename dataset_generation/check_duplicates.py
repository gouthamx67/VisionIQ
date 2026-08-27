from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CLEAN_DIR = (
    ROOT
    / "dataset"
    / "raw"
    / "clean"
)


def file_hash(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

        while chunk := file.read(
            1024 * 1024
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main():

    hashes = {}

    duplicates = []

    for path in sorted(
        CLEAN_DIR.rglob("*")
    ):

        if not path.is_file():
            continue

        digest = file_hash(path)

        if digest in hashes:

            duplicates.append(
                (
                    path,
                    hashes[digest],
                )
            )

        else:
            hashes[digest] = path

    print(
        f"Images checked: {len(hashes)}"
    )

    print(
        f"Duplicates found: {len(duplicates)}"
    )

    for duplicate, original in duplicates:

        print(
            f"DUPLICATE: {duplicate}"
        )

        print(
            f"ORIGINAL : {original}"
        )


if __name__ == "__main__":
    main()