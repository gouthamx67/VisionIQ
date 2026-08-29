import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from quality.decision import analyze_image
from vision.features import extract_features

DATA_DIRS = [
    os.path.join(ROOT, "data"),
    os.path.join(ROOT, "dataset"),
    os.path.join(ROOT, "datasets"),
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def find_images():
    found = []

    for base in DATA_DIRS:
        if not os.path.isdir(base):
            continue

        for root, _, files in os.walk(base):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext not in IMAGE_EXTS:
                    continue

                path = os.path.join(root, name)
                low = path.lower()

                for label in [
                    "motion_blur",
                    "compression",
                    "overexposure",
                    "underexposure",
                    "blur",
                    "noise",
                    "clean",
                    "none",
                ]:
                    if label in low:
                        found.append((path, label))
                        break

    return found


items = find_images()

if not items:
    raise SystemExit(
        "No labelled images found automatically. "
        "Run `find . -type f | grep -E '\\.(jpg|jpeg|png|webp)$' | head -50` "
        "and send me the output."
    )

print(f"Found {len(items)} labelled images")

rows = []

for path, true_label in items:
    try:
        features = extract_features(path)

        rows.append({
            "path": path,
            "true": true_label,
            "rf_degraded_prob": float(features.get("degraded_prob", 0.0)),
            "cv_candidates": str(features),
        })
    except Exception as e:
        print("SKIP:", path, e)

df = pd.DataFrame(rows)

if df.empty:
    raise SystemExit("Could not extract features.")

print("\nStage-1 probability distribution:")
print(df.groupby("true")["rf_degraded_prob"].describe())

print("\nGate sweep:")
print("=" * 80)

results = []

for threshold in np.arange(0.30, 0.851, 0.025):
    predicted = []

    for _, row in df.iterrows():
        predicted.append(
            "degraded"
            if row["rf_degraded_prob"] >= threshold
            else "clean"
        )

    true_binary = [
        "clean" if x in {"clean", "none"} else "degraded"
        for x in df["true"]
    ]

    acc = accuracy_score(true_binary, predicted)

    results.append({
        "threshold": round(float(threshold), 3),
        "accuracy": round(float(acc), 4),
    })

result_df = pd.DataFrame(results)

print(
    result_df.sort_values(
        "accuracy",
        ascending=False
    ).head(10).to_string(index=False)
)

best = result_df.loc[result_df["accuracy"].idxmax()]

print("\n" + "=" * 80)
print("BEST GATE")
print("=" * 80)
print(f"threshold = {best['threshold']}")
print(f"accuracy  = {best['accuracy']}")

print("\nPer-class Stage-1 probabilities:")
for label, group in df.groupby("true"):
    print(
        f"{label:16s} "
        f"min={group.rf_degraded_prob.min():.3f} "
        f"median={group.rf_degraded_prob.median():.3f} "
        f"max={group.rf_degraded_prob.max():.3f}"
    )

print("\nIMPORTANT:")
print(
    "This script ONLY measures the gate. "
    "It does not modify your source files or models."
)
