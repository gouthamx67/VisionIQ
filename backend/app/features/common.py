import cv2
import numpy as np

def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")

def validate_image_array(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")
    if image.size == 0:
        raise ValueError("Image array is empty.")
    if image.ndim not in (2, 3):
        raise ValueError(f"Unsupported image dimensions: {image.ndim}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}")
