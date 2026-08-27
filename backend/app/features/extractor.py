import numpy as np
from .common import validate_image_array
from .schema import FEATURE_NAMES
from .sharpness import sharpness_features
from .exposure import exposure_features
from .noise import noise_features
from .contrast import contrast_features
from .texture import texture_features

def extract_all_features(image: np.ndarray) -> dict[str, float]:
    validate_image_array(image)
    
    features = {}
    features.update(sharpness_features(image))
    features.update(exposure_features(image))
    features.update(noise_features(image))
    features.update(contrast_features(image))
    features.update(texture_features(image))

    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise RuntimeError(f"Missing expected features: {missing}")

    return {name: float(features[name]) for name in FEATURE_NAMES}
