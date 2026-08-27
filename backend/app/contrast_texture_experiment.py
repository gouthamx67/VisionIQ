import cv2

from features.contrast import contrast_features
from features.texture import texture_features

# Dictionary of image names and their file paths
images = {
    "sharp": "../../data/samples/sharp.png",
    "slightly_blurred": "../../data/samples/slightly_blurred.png",
    "heavily_blurred": "../../data/samples/heavily_blurred.png",
    "noise_original": "../../data/samples/noise_original.png",
    "noise_low": "../../data/samples/noise_low.png",
    "noise_medium": "../../data/samples/noise_medium.png",
    "noise_high": "../../data/samples/noise_high.png",
}

for name, path in images.items():
    image = cv2.imread(path)

    if image is None:
        print(f"Could not load {name} at {path}. Skipping...")
        continue

    contrast = contrast_features(image)
    texture = texture_features(image)

    print()
    print("=" * 55)
    print(name.upper())
    print("=" * 55)

    print("CONTRAST FEATURES")
    for feature_name, value in contrast.items():
        print(f"{feature_name:30s}: {value:.4f}")

    print("\nTEXTURE FEATURES")
    for feature_name, value in texture.items():
        print(f"{feature_name:30s}: {value:.4f}")
