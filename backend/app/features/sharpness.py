import cv2
import numpy as np

from .common import to_grayscale, validate_image_array


def sharpness_features(image: np.ndarray) -> dict[str, float]:
    validate_image_array(image)
    gray = to_grayscale(image)
    gray_float = gray.astype(np.float32)

    laplacian = cv2.Laplacian(gray_float, cv2.CV_32F)
    laplacian_variance = float(laplacian.var())

    return {
        "laplacian_variance": laplacian_variance,
    }
