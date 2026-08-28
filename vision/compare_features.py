import sys
import os
# Add the parent directory (VisionIQ root) to the system path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import glob
from pathlib import Path
from vision.features import extract_features

# 1. Find the first clean image from the training split
clean_path = Path('dataset/raw/clean') / open('dataset/splits/train.txt').readline().strip()

# 2. Find the first generated image for each degradation type
def find_first(pattern):
    matches = glob.glob(f'dataset/generated/**/{pattern}', recursive=True)
    return matches[0] if matches else None

blur_path = find_first('*blur*')
noise_path = find_first('*noise*')
underexposure_path = find_first('*underexposure*')
overexposure_path = find_first('*overexposure*')

# 3. Bundle them together
images = {
    "Clean": clean_path,
    "Blur": blur_path,
    "Noise": noise_path,
    "Underexposure": underexposure_path,
    "Overexposure": overexposure_path,
}

# 4. Run the extractor on each
for name, path in images.items():
    if path is None:
        print(f"Could not find an example for {name}. Skipping...")
        continue

    image = cv2.imread(str(path))
    if image is None:
        print(f"Could not load image at {path}")
        continue

    print(f"\n{'='*60}")
    print(f"FEATURES FOR: {name.upper()}")
    print(f"File: {path}")
    print(f"{'='*60}")

    features = extract_features(image)
    
    # Print the most important features for this task
    # (You can print all of them if you want!)
    for key, value in features.items():
        if key in ["width", "height", "aspect_ratio"]:
            continue # Skip metadata for a cleaner look
        print(f"{key:30s}: {value:.4f}")

print("\nDone!")
