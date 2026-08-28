import sys
import os
# Add parent directory (VisionIQ root) to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quality.decision import analyze_image

# Use a clean image from your test split
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_image_path = os.path.join(ROOT, "dataset/raw/clean", open(os.path.join(ROOT, "dataset/splits/test.txt")).readline().strip())

print("Analyzing image:", test_image_path)
result = analyze_image(test_image_path)
print(result)
