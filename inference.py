import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from quality.decision import analyze_image

def analyze_folder(folder_path):
    print(f"Analyzing folder: {folder_path}")
    count = 0
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            image_path = os.path.join(folder_path, filename)
            try:
                result = analyze_image(image_path)
                print(f"{filename}: Score={result['quality_score']}, Label={result['quality_label']}, Issues={result['issues']}")
                count += 1
            except Exception as e:
                print(f"ERROR on {filename}: {e}")
    print(f"Done! Analyzed {count} images.")

if __name__ == "__main__":
    # Analyze the test set (clean images)
    analyze_folder("dataset/raw/clean")
    # Analyze the generated test set (degraded images)
    analyze_folder("dataset/generated/test")
