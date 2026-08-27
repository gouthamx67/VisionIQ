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
    )

    histogram = histogram.flatten()

    probabilities = histogram / histogram.sum()

    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    return float(entropy)


def noise_features(image: np.ndarray) -> dict[str, float]:
    """
    Extract noise-related features from an image.

    These measurements provide evidence about random
    high-frequency variation but do not independently
    determine whether an image is noisy.
    """

    if image.ndim == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = image

    gray_float = gray.astype(np.float32)

    # -----------------------------------------------------
    # Global variation
    # -----------------------------------------------------

    grayscale_std = float(
        gray_float.std()
    )

    # -----------------------------------------------------
    # Local variance
    # -----------------------------------------------------

    local_mean = cv2.GaussianBlur(
        gray_float,
        (5, 5),
        0,
    )

    local_squared_mean = cv2.GaussianBlur(
        gray_float ** 2,
        (5, 5),
        0,
    )

    local_variance = (
        local_squared_mean
        - local_mean ** 2
    )

    local_variance_mean = float(
        np.mean(local_variance)
    )

    # -----------------------------------------------------
    # High-frequency residual
    # -----------------------------------------------------

    blurred = cv2.GaussianBlur(
        gray_float,
        (5, 5),
        0,
    )

    residual = (
        gray_float - blurred
    )

    residual_mean = float(
        residual.mean()
    )

    residual_std = float(
        residual.std()
    )

    residual_abs_mean = float(
        np.mean(np.abs(residual))
    )

    # -----------------------------------------------------
    # Entropy
    # -----------------------------------------------------

    entropy = calculate_entropy(gray)

    return {
        "grayscale_std": grayscale_std,
        "local_variance_mean": local_variance_mean,
        "high_frequency_mean": residual_mean,
        "high_frequency_std": residual_std,
        "high_frequency_abs_mean": residual_abs_mean,
        "entropy": entropy,
    }