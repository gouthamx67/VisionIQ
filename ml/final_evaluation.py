import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_PATH = os.path.join(ROOT, "dataset/metadata/feature_dataset.csv")
ARTIFACT_DIR = os.path.join(ROOT, "ml/artifacts")

df = pd.read_csv(FEATURE_PATH)
test_df = df[df["split"] == "test"]

EXCLUDED_COLUMNS = ["source_id", "split", "variant_id", "degradation", "severity", "severity_name", "is_degraded"]
features = [c for c in df.columns if c not in EXCLUDED_COLUMNS]

X_test = test_df[features]

# Load Stage 1 and Stage 2 models
clf1 = joblib.load(os.path.join(ARTIFACT_DIR, "stage1_clean_vs_degraded.joblib"))
le1 = joblib.load(os.path.join(ARTIFACT_DIR, "stage1_label_encoder.joblib"))
clf2 = joblib.load(os.path.join(ARTIFACT_DIR, "stage2_degradation_type.joblib"))
le2 = joblib.load(os.path.join(ARTIFACT_DIR, "stage2_label_encoder.joblib"))

print("="*60)
print("FINAL EVALUATION ON TEST SET")
print("="*60)

# Stage 1 Metrics
y1_true = le1.transform(test_df["is_degraded"])
y1_pred = clf1.predict(X_test)
print("\nSTAGE 1: Clean vs Degraded")
print(f"Accuracy: {accuracy_score(y1_true, y1_pred):.4f}")
print(classification_report(y1_true, y1_pred, target_names=["Clean", "Degraded"]))

# Stage 2 Metrics
deg_indices = test_df[test_df["is_degraded"] == 1].index
X_deg = X_test.loc[deg_indices]
y2_true = le2.transform(test_df.loc[deg_indices, "degradation"])
y2_pred = clf2.predict(X_deg)
print("\nSTAGE 2: Degradation Type (on Degraded Images)")
print(f"Accuracy: {accuracy_score(y2_true, y2_pred):.4f}")
print(classification_report(y2_true, y2_pred, target_names=le2.classes_))
print("Confusion Matrix:\n", confusion_matrix(y2_true, y2_pred))

# Overall Macro F1
macro_f1 = classification_report(y2_true, y2_pred, output_dict=True)["macro avg"]["f1-score"]
print(f"\nOverall Macro F1 (Stage 2): {macro_f1:.4f}")
