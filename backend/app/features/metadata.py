import cv2
import numpy as np


def get_image_metadata(image: np.ndarray) -> dict:
    """
    Extract basic metadata about the image.
    This is NOT used as an ML feature, but is useful for the API response.
    """
    
    height, width = image.shape[:2]
    
    if image.ndim == 3:
        channels = image.shape[2]
    else:
        channels = 1  # Grayscale image
        
    return {
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
    }
