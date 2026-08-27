import cv2
from features.extractor import extract_all_features
from features.metadata import get_image_metadata

IMAGE_PATH = "../../data/samples/sharp.png"

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise RuntimeError(f"Could not load image: {IMAGE_PATH}")

# --- Extract ML Features ---
features = extract_all_features(image)

# --- Extract Metadata (Not ML features) ---
metadata = get_image_metadata(image)


print()
print("=" * 70)
print("VISIONIQ FEATURE VECTOR")
print("=" * 70)

for name, value in features.items():
    print(f"{name:35s}: {value:.6f}")

print()
print("Number of features:", len(features))


print()
print("=" * 70)
print("IMAGE METADATA")
print("=" * 70)

for name, value in metadata.items():
    print(f"{name:35s}: {value}")

