import cv2
import numpy as np


def texture_features(
    image: np.ndarray,
) -> dict[str, float]:
    """
    Extract simple texture and spatial-structure features.
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
    # Gradient-based texture
    # -----------------------------------------------------

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

    gradient_magnitude = cv2.magnitude(
        gradient_x,
        gradient_y,
    )

    gradient_mean = float(
        gradient_magnitude.mean()
    )

    gradient_std = float(
        gradient_magnitude.std()
    )

    # -----------------------------------------------------
    # Texture strength
    # -----------------------------------------------------

    texture_energy = float(
        np.mean(
            gradient_magnitude ** 2
        )
    )

    # -----------------------------------------------------
    # High-gradient proportion
    # -----------------------------------------------------

    texture_threshold = 25.0

    textured_pixel_ratio = float(
        np.mean(
            gradient_magnitude
            >= texture_threshold
        )
    )

    return {
        "texture_gradient_mean": gradient_mean,
        "texture_gradient_std": gradient_std,
        "texture_energy": texture_energy,
        "textured_pixel_ratio": textured_pixel_ratio,
    }
