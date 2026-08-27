import cv2
import numpy as np

from .common import to_grayscale, validate_image_array


def exposure_features(image: np.ndarray) -> dict[str, float]:
    validate_image_array(image)
    gray = to_grayscale(image)
    gray_float = gray.astype(np.float32)

    mean_brightness = float(gray_float.mean())
    brightness_std = float(gray_float.std())

    percentile_1 = float(np.percentile(gray_float, 1))
    percentile_50 = float(np.percentile(gray_float, 50))
    percentile_99 = float(np.percentile(gray_float, 99))
    dynamic_range = percentile_99 - percentile_1

    dark_ratio = float(np.mean(gray_float < 50))
    bright_ratio = float(np.mean(gray_float > 200))

    black_clip_ratio = float(np.mean(gray_float < 5))
    white_clip_ratio = float(np.mean(gray_float > 250))

    return {
        "mean_brightness": mean_brightness,
        "brightness_std": brightness_std,
        "dynamic_range": dynamic_range,
        "percentile_1": percentile_1,
        "percentile_50": percentile_50,
        "percentile_99": percentile_99,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "black_clip_ratio": black_clip_ratio,
        "white_clip_ratio": white_clip_ratio,
    }
