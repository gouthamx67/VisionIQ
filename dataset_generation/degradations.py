from __future__ import annotations

import cv2
import numpy as np


def gaussian_blur(
    image: np.ndarray,
    severity: int,
) -> np.ndarray:

    kernels = {
        1: 3,
        2: 5,
        3: 9,
        4: 13,
        5: 21,
    }

    kernel = kernels[severity]

    return cv2.GaussianBlur(
        image,
        (kernel, kernel),
        0,
    )


def motion_blur(
    image: np.ndarray,
    severity: int,
) -> np.ndarray:

    sizes = {
        1: 5,
        2: 9,
        3: 15,
        4: 21,
        5: 31,
    }

    size = sizes[severity]

    kernel = np.zeros(
        (size, size),
        dtype=np.float32,
    )

    kernel[size // 2, :] = 1.0 / size

    return cv2.filter2D(
        image,
        -1,
        kernel,
    )


def adjust_exposure(
    image: np.ndarray,
    severity: int,
    direction: str,
) -> np.ndarray:

    under_factors = {
        1: 0.85,
        2: 0.70,
        3: 0.55,
        4: 0.40,
        5: 0.25,
    }

    over_factors = {
        1: 1.15,
        2: 1.30,
        3: 1.50,
        4: 1.75,
        5: 2.00,
    }

    if direction == "underexposure":
        factor = under_factors[severity]

    elif direction == "overexposure":
        factor = over_factors[severity]

    else:
        raise ValueError(
            "direction must be "
            "'underexposure' or "
            "'overexposure'"
        )

    output = image.astype(
        np.float32
    ) * factor

    return np.clip(
        output,
        0,
        255,
    ).astype(np.uint8)


def gaussian_noise(
    image: np.ndarray,
    severity: int,
    rng: np.random.Generator,
) -> np.ndarray:

    sigmas = {
        1: 5,
        2: 10,
        3: 20,
        4: 30,
        5: 45,
    }

    sigma = sigmas[severity]

    noise = rng.normal(
        0,
        sigma,
        image.shape,
    )

    noisy = (
        image.astype(np.float32)
        + noise
    )

    return np.clip(
        noisy,
        0,
        255,
    ).astype(np.uint8)


def jpeg_compression(
    image: np.ndarray,
    severity: int,
) -> np.ndarray:

    qualities = {
        1: 90,
        2: 75,
        3: 50,
        4: 30,
        5: 10,
    }

    quality = qualities[severity]

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            quality,
        ],
    )

    if not success:
        raise RuntimeError(
            "JPEG encoding failed."
        )

    decoded = cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR,
    )

    if decoded is None:
        raise RuntimeError(
            "JPEG decoding failed."
        )

    return decoded