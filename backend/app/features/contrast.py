import cv2
import numpy as np


def contrast_features(
    image: np.ndarray,
) -> dict[str, float]:
    """
    Extract global and local contrast features.
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
    # Global contrast
    # -----------------------------------------------------

    global_std = float(
        gray_float.std()
    )

    percentile_1 = np.percentile(
        gray_float,
        1,
    )

    percentile_99 = np.percentile(
        gray_float,
        99,
    )

    percentile_contrast = float(
        percentile_99 - percentile_1
    )

    # -----------------------------------------------------
    # Local contrast
    # -----------------------------------------------------

    local_mean = cv2.GaussianBlur(
        gray_float,
        (7, 7),
        0,
    )

    local_squared_mean = cv2.GaussianBlur(
        gray_float ** 2,
        (7, 7),
        0,
    )

    local_variance = (
        local_squared_mean
        - local_mean ** 2
    )

    local_variance = np.maximum(
        local_variance,
        0,
    )

    local_std = np.sqrt(
        local_variance
    )

    local_contrast_mean = float(
        local_std.mean()
    )

    # -----------------------------------------------------
    # Gradient statistics
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

    strong_gradient_threshold = 50.0

    strong_gradient_ratio = float(
        np.mean(
            gradient_magnitude
            >= strong_gradient_threshold
        )
    )

    return {
        "global_contrast": global_std,
        "percentile_contrast": percentile_contrast,
        "local_contrast_mean": local_contrast_mean,
        "gradient_mean": gradient_mean,
        "gradient_std": gradient_std,
        "strong_gradient_ratio": strong_gradient_ratio,
    }
