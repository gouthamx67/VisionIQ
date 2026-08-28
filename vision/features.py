from __future__ import annotations

import cv2
import numpy as np


def calculate_entropy(gray: np.ndarray) -> float:
    """
    Calculate Shannon entropy of a grayscale image.
    """
    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256],
    ).flatten()

    probabilities = (
        histogram
        / histogram.sum()
    )

    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = -np.sum(
        probabilities
        * np.log2(probabilities)
    )

    return float(entropy)


def calculate_noise_estimate(
    gray: np.ndarray,
) -> float:
    """
    Estimate high-frequency residual energy.
    This is a heuristic noise indicator,
    not a physical sensor-noise measurement.
    """
    smooth = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    residual = (
        gray.astype(np.float32)
        - smooth.astype(np.float32)
    )

    return float(
        np.std(residual)
    )


def calculate_gradient_features(
    gray: np.ndarray,
) -> tuple[float, float]:

    gray_float = gray.astype(
        np.float32
    )

    gradient_x = cv2.Sobel(
        gray_float,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gradient_y = cv2.Sobel(
        gray_float,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude = cv2.magnitude(
        gradient_x,
        gradient_y,
    )

    mean_gradient = float(
        np.mean(magnitude)
    )

    std_gradient = float(
        np.std(magnitude)
    )

    return (
        mean_gradient,
        std_gradient,
    )


def calculate_texture_variation(
    gray: np.ndarray,
) -> float:
    """
    Estimate local texture variation.
    """
    local_mean = cv2.GaussianBlur(
        gray.astype(np.float32),
        (7, 7),
        0,
    )

    squared_difference = (
        gray.astype(np.float32)
        - local_mean
    ) ** 2

    local_variance = cv2.GaussianBlur(
        squared_difference,
        (7, 7),
        0,
    )

    return float(
        np.mean(
            np.sqrt(
                local_variance
            )
        )
    )


def extract_features(
    image: np.ndarray,
) -> dict[str, float]:

    if image is None:
        raise ValueError(
            "Image cannot be None."
        )

    if image.ndim != 3:
        raise ValueError(
            "Expected a color image."
        )

    if image.shape[2] != 3:
        raise ValueError(
            "Expected 3-channel image."
        )

    height, width = image.shape[:2]

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    # -------------------------------------------------
    # Brightness
    # -------------------------------------------------

    mean_brightness = float(
        np.mean(gray)
    )

    brightness_std = float(
        np.std(gray)
    )

    # -------------------------------------------------
    # Exposure
    # -------------------------------------------------

    dark_ratio = float(
        np.mean(gray < 40)
    )

    bright_ratio = float(
        np.mean(gray > 215)
    )

    black_clip_ratio = float(
        np.mean(gray == 0)
    )

    white_clip_ratio = float(
        np.mean(gray == 255)
    )

    # -------------------------------------------------
    # Percentiles
    # -------------------------------------------------

    p01, p05, p50, p95, p99 = (
        np.percentile(
            gray,
            [1, 5, 50, 95, 99],
        )
    )

    contrast_range = float(
        p95 - p05
    )

    dynamic_range = float(
        p99 - p01
    )

    # -------------------------------------------------
    # Sharpness
    # -------------------------------------------------

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    )

    laplacian_variance = float(
        laplacian.var()
    )

    (
        mean_gradient,
        std_gradient,
    ) = calculate_gradient_features(
        gray
    )

    # -------------------------------------------------
    # Saturation
    # -------------------------------------------------

    saturation = hsv[:, :, 1]

    mean_saturation = float(
        np.mean(saturation)
    )

    high_saturation_ratio = float(
        np.mean(
            saturation > 200
        )
    )

    low_saturation_ratio = float(
        np.mean(
            saturation < 25
        )
    )

    # -------------------------------------------------
    # Noise
    # -------------------------------------------------

    noise_estimate = (
        calculate_noise_estimate(
            gray
        )
    )

    # -------------------------------------------------
    # Entropy
    # -------------------------------------------------

    entropy = calculate_entropy(
        gray
    )

    # -------------------------------------------------
    # Texture
    # -------------------------------------------------

    texture_variation = (
        calculate_texture_variation(
            gray
        )
    )

    # -------------------------------------------------
    # Geometry
    # -------------------------------------------------

    aspect_ratio = (
        width / height
    )

    # -------------------------------------------------
    # Return feature vector
    # -------------------------------------------------

    return {
        "width": float(width),
        "height": float(height),
        "aspect_ratio": float(
            aspect_ratio
        ),
        "mean_brightness": (
            mean_brightness
        ),
        "brightness_std": (
            brightness_std
        ),
        "dark_ratio": (
            dark_ratio
        ),
        "bright_ratio": (
            bright_ratio
        ),
        "black_clip_ratio": (
            black_clip_ratio
        ),
        "white_clip_ratio": (
            white_clip_ratio
        ),
        "p01": float(p01),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
        "contrast_range": (
            contrast_range
        ),
        "dynamic_range": (
            dynamic_range
        ),
        "laplacian_variance": (
            laplacian_variance
        ),
        "mean_gradient": (
            mean_gradient
        ),
        "std_gradient": (
            std_gradient
        ),
        "mean_saturation": (
            mean_saturation
        ),
        "high_saturation_ratio": (
            high_saturation_ratio
        ),
        "low_saturation_ratio": (
            low_saturation_ratio
        ),
        "noise_estimate": (
            noise_estimate
        ),
        "entropy": (
            entropy
        ),
        "texture_variation": (
            texture_variation
        ),
    }
