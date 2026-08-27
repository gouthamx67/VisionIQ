import cv2
import numpy as np


def calculate_laplacian_variance(image: np.ndarray) -> float:
    """
    Calculate the variance of the Laplacian response.

    Higher values generally indicate stronger high-frequency
    detail and therefore greater apparent sharpness.
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return float(laplacian.var())


# ---------------------------------------------------------
# 1. Create a synthetic image with strong edges
# ---------------------------------------------------------

image = np.full((512, 512), 128, dtype=np.uint8)

cv2.rectangle(
    image,
    (100, 100),
    (412, 412),
    0,
    -1,
)

cv2.circle(
    image,
    (256, 256),
    80,
    255,
    -1,
)


# ---------------------------------------------------------
# 2. Create different levels of blur
# ---------------------------------------------------------

slightly_blurred = cv2.GaussianBlur(
    image,
    (9, 9),
    0,
)

heavily_blurred = cv2.GaussianBlur(
    image,
    (31, 31),
    0,
)


# ---------------------------------------------------------
# 3. Calculate sharpness measurements
# ---------------------------------------------------------

sharpness_original = calculate_laplacian_variance(image)
sharpness_mild = calculate_laplacian_variance(slightly_blurred)
sharpness_heavy = calculate_laplacian_variance(heavily_blurred)


# ---------------------------------------------------------
# 4. Print results
# ---------------------------------------------------------

print("Laplacian variance")
print("------------------")

print(f"Original:       {sharpness_original:.2f}")
print(f"Slightly blur:  {sharpness_mild:.2f}")
print(f"Heavily blur:   {sharpness_heavy:.2f}")


# ---------------------------------------------------------
# 5. Save images for visual inspection
# ---------------------------------------------------------

cv2.imwrite(
    "data/samples/sharp.png",
    image,
)

cv2.imwrite(
    "data/samples/slightly_blurred.png",
    slightly_blurred,
)

cv2.imwrite(
    "data/samples/heavily_blurred.png",
    heavily_blurred,
)

print()
print("Saved:")
print("  data/samples/sharp.png")
print("  data/samples/slightly_blurred.png")
print("  data/samples/heavily_blurred.png")