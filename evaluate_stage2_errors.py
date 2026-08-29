import cv2
import pandas as pd
import joblib

from vision.features import extract_features
from quality.decision import _feature_frame, _cv_candidates


df = pd.read_csv("dataset/metadata/feature_dataset.csv")

test = df[
    (df["split"] == "test") &
    (df["is_degraded"] == 1)
].copy()

model = joblib.load(
    "ml/artifacts/stage2_degradation_type.joblib"
)

encoder = joblib.load(
    "ml/artifacts/stage2_label_encoder.joblib"
)

rows = []

for _, row in test.iterrows():

    path = (
        f"dataset/generated/{row['split']}/"
        f"{row['variant_id']}.jpg"
    )

    image = cv2.imread(path)

    if image is None:
        continue

    features = extract_features(image)

    X = _feature_frame(features)
    X = X.reindex(columns=model.feature_names_in_)

    probs = model.predict_proba(X)[0]
    pred_idx = int(probs.argmax())

    predicted = encoder.classes_[pred_idx]

    cv_candidates = _cv_candidates(features)

    rows.append({
        "id": row["variant_id"],
        "true": row["degradation"],
        "predicted": predicted,
        "confidence": float(probs[pred_idx]),
        "cv_candidates": ",".join(cv_candidates) if cv_candidates else "none",

        "brightness": features.get("mean_brightness", 0),
        "dark_ratio": features.get("dark_ratio", 0),
        "bright_ratio": features.get("bright_ratio", 0),
        "white_clip": features.get("white_clip_ratio", 0),
        "black_clip": features.get("black_clip_ratio", 0),

        "laplacian": features.get("laplacian_variance", 0),
        "gradient": features.get("mean_gradient", 0),
        "noise": features.get("noise_estimate", 0),
        "entropy": features.get("entropy", 0),
        "texture": features.get("texture_variation", 0),
    })


out = pd.DataFrame(rows)

print()
print("=" * 110)
print("VISIONIQ STAGE 2 RAW MODEL DIAGNOSTIC")
print("=" * 110)

print()
print("EVALUATED:", len(out))

print()
print("RAW STAGE 2 CONFUSION:")
print(
    pd.crosstab(
        out["true"],
        out["predicted"],
        margins=True,
    ).to_string()
)

accuracy = (
    out["true"] == out["predicted"]
).mean()

print()
print("RAW STAGE 2 ACCURACY:")
print(round(float(accuracy), 4))

print()
print("PER-CLASS ACCURACY:")

for cls in sorted(out["true"].unique()):

    subset = out[out["true"] == cls]

    correct = (
        subset["true"] == subset["predicted"]
    ).sum()

    cls_acc = correct / len(subset)

    print(
        f"{cls:15s} "
        f"{correct:2d}/{len(subset):2d} "
        f"({cls_acc:.3f})"
    )

print()
print("=" * 110)
print("STAGE 2 ERRORS")
print("=" * 110)

errors = out[
    out["true"] != out["predicted"]
]

print("Total errors:", len(errors))

print()
print("ERROR COUNTS:")
print(
    pd.crosstab(
        errors["true"],
        errors["predicted"],
    ).to_string()
)

print()
print("=" * 110)
print("MOTION BLUR")
print("=" * 110)

print(
    out[out["true"] == "motion_blur"]
    .sort_values("laplacian")
    .to_string(index=False)
)

print()
print("=" * 110)
print("COMPRESSION")
print("=" * 110)

print(
    out[out["true"] == "compression"]
    .sort_values("laplacian")
    .to_string(index=False)
)

print()
print("=" * 110)
print("CV CANDIDATE COUNTS")
print("=" * 110)

print(
    out["cv_candidates"]
    .value_counts()
    .to_string()
)
