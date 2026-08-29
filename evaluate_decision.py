import os
import pandas as pd

from quality.decision import analyze_image


df = pd.read_csv(
    "dataset/metadata/feature_dataset.csv"
)

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

    try:
        result = analyze_image(path)

        # Hierarchical classifier:
        # Stage 2 is authoritative when it runs.
        # Do not reconstruct the diagnosis from the issue list.
        stage2_prediction = result.get("stage2_prediction")
        stage2_confidence = result.get("stage2_confidence")

        if stage2_prediction is not None:
            final_prediction = stage2_prediction
            final_confidence = float(stage2_confidence or 0.0)
        else:
            final_prediction = "none"
            final_confidence = 1.0

        rows.append(
            {
                "id": row["variant_id"],
                "true": row["degradation"],

                "stage1_prob": result.get(
                    "stage1_degraded_prob"
                ),

                "stage2_prediction": result.get(
                    "stage2_prediction"
                ),

                "stage2_confidence": result.get(
                    "stage2_confidence"
                ),

                "cv_candidates": ",".join(
                    result.get(
                        "cv_candidates",
                        []
                    )
                ),

                "final": final_prediction,

                "final_confidence": final_confidence,
            }
        )

    except Exception as e:
        print("ERROR:", path, e)


out = pd.DataFrame(rows)

print("\n" + "=" * 110)
print("VISIONIQ STAGE 2 DIAGNOSTIC")
print("=" * 110)

print("\nEVALUATED:", len(out))

print("\nSTAGE 2 RF CONFUSION:")
print(
    pd.crosstab(
        out["true"],
        out["stage2_prediction"],
        margins=True,
    ).to_string()
)

stage2_valid = out["stage2_prediction"].notna()

if stage2_valid.any():

    stage2_accuracy = (
        out.loc[stage2_valid, "true"]
        == out.loc[stage2_valid, "stage2_prediction"]
    ).mean()

    print("\nSTAGE 2 RF ACCURACY:")
    print(round(float(stage2_accuracy), 4))


print("\nFINAL CONFUSION:")
print(
    pd.crosstab(
        out["true"],
        out["final"],
        margins=True,
    ).to_string()
)

final_accuracy = (
    out["true"] == out["final"]
).mean()

print("\nFINAL ACCURACY:")
print(round(float(final_accuracy), 4))


print("\n" + "=" * 110)
print("MOTION BLUR DIAGNOSTIC")
print("=" * 110)

print(
    out[out["true"] == "motion_blur"]
    .to_string(index=False)
)


print("\n" + "=" * 110)
print("COMPRESSION DIAGNOSTIC")
print("=" * 110)

print(
    out[out["true"] == "compression"]
    .to_string(index=False)
)


print("\n" + "=" * 110)
print("CV OVERRIDE IMPACT")
print("=" * 110)

override = (
    out["stage2_prediction"].notna()
    & (
        out["stage2_prediction"]
        != out["final"]
    )
)

print(
    "Stage 2 predictions changed by decision layer:",
    int(override.sum()),
    "/",
    len(out),
)

print("\nFINAL CONFIDENCE:")
print(
    round(
        float(
            out["final_confidence"].mean()
        ),
        4,
    )
)
