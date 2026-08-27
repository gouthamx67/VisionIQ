import cv2
import numpy as np
import os

from features.noise import noise_features


# ---------------------------------------------------------
# Load original image
# ---------------------------------------------------------

image = cv2.imread(
    "../../data/samples/sharp.png",
)

if image is None:
    raise RuntimeError(
        "Could not load test image. Please ensure 'sharp.png' is in the VisionIQ/data/samples folder."
    )


# ---------------------------------------------------------
# Convert to grayscale
# ---------------------------------------------------------

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY,
)


# ---------------------------------------------------------
# Generate controlled Gaussian noise
# ---------------------------------------------------------

rng = np.random.default_rng(42)


noise_low = rng.normal(
    loc=0,
    scale=5,
    size=gray.shape,
)

noise_medium = rng.normal(
    loc=0,
    scale=15,
    size=gray.shape,
)

noise_high = rng.normal(
    loc=0,
    scale=30,
    size=gray.shape,
)


# ---------------------------------------------------------
# Add noise
# ---------------------------------------------------------

noisy_low = np.clip(
    gray.astype(np.float32)
    + noise_low,
    0,
    255,
).astype(np.uint8)


noisy_medium = np.clip(
    gray.astype(np.float32)
    + noise_medium,
    0,
    255,
).astype(np.uint8)


noisy_high = np.clip(
    gray.astype(np.float32)
    + noise_high,
    0,
    255,
).astype(np.uint8)


images = {
    "original": gray,
    "low_noise": noisy_low,
    "medium_noise": noisy_medium,
    "high_noise": noisy_high,
}


# ---------------------------------------------------------
# Analyze
# ---------------------------------------------------------

for name, current_image in images.items():

    features = noise_features(
        current_image
    )

    print()
    print("=" * 55)
    print(name.upper())
    print("=" * 55)

    for feature_name, value in features.items():

        print(
            f"{feature_name:25s}: "
            f"{value:.4f}"
        )


# ---------------------------------------------------------
# Save (Updated to match the read path)
# ---------------------------------------------------------

# Ensure the output directory exists
os.makedirs("../../data/samples", exist_ok=True)

cv2.imwrite(
    "../../data/samples/noise_original.png",
    gray,
)

cv2.imwrite(
    "../../data/samples/noise_low.png",
    noisy_low,
)

cv2.imwrite(
    "../../data/samples/noise_medium.png",
    noisy_medium,
)

cv2.imwrite(
    "../../data/samples/noise_high.png",
    noisy_high,
)


print()
print("Noise experiment images saved.")