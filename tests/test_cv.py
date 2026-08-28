import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from vision.features import extract_features

def test_extract_features_returns_dict():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    features = extract_features(img)
    assert isinstance(features, dict)
    assert "laplacian_variance" in features
    assert "mean_brightness" in features

def test_blur_reduces_laplacian():
    img = cv2.imread("dataset/raw/clean/" + open("dataset/splits/train.txt").readline().strip())
    sharp = extract_features(img)["laplacian_variance"]
    blurred = cv2.GaussianBlur(img, (21, 21), 0)
    blur_features = extract_features(blurred)["laplacian_variance"]
    assert blur_features < sharp
