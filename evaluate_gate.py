import os
import pandas as pd
import cv2

from quality.decision import (
    extract_features,
    _feature_frame,
    clf_stage1,
    le_stage1,
    clf_stage2,
    le_stage2,
    _cv_candidates,
)

df = pd.read_csv("dataset/metadata/feature_dataset.csv")
test = df[df["split"] == "test"].copy()

rows = []

for _, row in test.iterrows():

    path = os.path.join(
        "dataset/generated",
        row["split"],
        row["variant_id"] + ".jpg",
    )

    if not os.path.exists(path):
        continue

    image = cv2.imread(path)

    if image is None:
        continue

    features = extract_features(image)

    X = _feature_frame(features)
    X = X.reindex(columns=clf_stage1.feature_names_in_)

    stage1_probs = clf_stage1.predict_proba(X)[0]

    class_1_idx = list(le_stage1.classes_).index(1)

    stage1_degraded_prob = float(
        stage1_probs[class_1_idx]
    )

    stage2_probs = clf_stage2.predict_proba(X)[0]

    stage2_idx = stage2_probs.argmax()

    stage2_prediction = le_stage2.inverse_transform(
        [stage2_idx]
    )[0]

    stage2_confidence = float(
        stage2_probs[stage2_idx]
    )

    cv = _cv_candidates(features)

    rows.append({
        "true": row["degradation"],
        "stage1_degraded_prob": stage1_degraded_prob,
        "stage2_prediction": stage2_prediction,
        "stage2_confidence": stage2_confidence,
        "cv_candidates": ",".join(cv) if cv else "none",
    })

out = pd.DataFrame(rows)

print()
print("=" * 110)
print("VISIONIQ GATE DIAGNOSTIC")
print("=" * 110)

print("\nEVALUATED:", len(out))

print("\nSTAGE 1 DEGRADATION PROBABILITY BY CLASS:")

print(
    out.groupby("true")["stage1_degraded_prob"]
       .agg(["min", "median", "max"])
       .round(3)
       .to_string()
)

print("\n" + "=" * 110)
print("STAGE 1 ACCEPTANCE @ 0.70")
print("=" * 110)

out["stage1_accept"] = (
    out["stage1_degraded_prob"] >= 0.70
)

for cls in sorted(out["true"].unique()):

    subset = out[out["true"] == cls]

    accepted = int(subset["stage1_accept"].sum())

    print(
        f"{cls:15s} "
        f"{accepted:2d}/{len(subset):2d} "
        f"accepted "
        f"({accepted / len(subset):.3f})"
    )

print("\n" + "=" * 110)
print("STAGE 2 IF WE IGNORE THE GATE")
print("=" * 110)

degraded = out[
    out["true"] != "none"
].copy()

print(
    pd.crosstab(
        degraded["true"],
        degraded["stage2_prediction"],
        margins=True,
    ).to_string()
)

print("\n" + "=" * 110)
print("STAGE 2 CONFIDENCE BY TRUE CLASS")
print("=" * 110)

print(
    out.groupby("true")["stage2_confidence"]
       .agg(["min", "median", "max"])
       .round(3)
       .to_string()
)
