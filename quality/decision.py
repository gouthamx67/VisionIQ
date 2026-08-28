from __future__ import annotations

import os
import sys
import cv2
import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vision.features import extract_features

ARTIFACT_DIR = os.path.join(ROOT, "ml", "artifacts")
clf_stage1 = joblib.load(os.path.join(ARTIFACT_DIR, "stage1_clean_vs_degraded.joblib"))
le_stage1 = joblib.load(os.path.join(ARTIFACT_DIR, "stage1_label_encoder.joblib"))
clf_stage2 = joblib.load(os.path.join(ARTIFACT_DIR, "stage2_degradation_type.joblib"))
le_stage2 = joblib.load(os.path.join(ARTIFACT_DIR, "stage2_label_encoder.joblib"))

CSV_PATH = os.path.join(ROOT, "dataset/metadata/feature_dataset.csv")
df_cols = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()
EXCLUDED_COLUMNS = ["source_id", "split", "variant_id", "degradation", "severity", "severity_name", "is_degraded"]
FEATURE_NAMES = [c for c in df_cols if c not in EXCLUDED_COLUMNS]

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))

def confidence_from_probability(probability):
    return round(clamp(probability, 0.0, 1.0), 3)

def calculate_quality_score(issue_scores):
    if not issue_scores:
        return 100.0
    total_penalty = 0.0
    weights = {"blur": 25.0, "motion_blur": 25.0, "underexposure": 20.0, "overexposure": 20.0, "noise": 20.0, "compression": 20.0}
    for issue_type, severity_score in issue_scores.items():
        weight = weights.get(issue_type, 15.0)
        severity_score = clamp(severity_score, 0.0, 1.0)
        total_penalty += (weight * severity_score)
    return round(clamp(100.0 - total_penalty, 0.0, 100.0), 2)

def quality_label_from_score(score):
    if score >= 80: return "ACCEPTABLE"
    if score >= 50: return "DEGRADED"
    return "POTENTIALLY_DEFECTIVE"

def analyze_image(image_path):
    image = cv2.imread(image_path)
    if image is None: raise RuntimeError(f"Could not load image: {image_path}")

    features = extract_features(image)
    X = pd.DataFrame([[float(features.get(name, 0.0)) for name in FEATURE_NAMES]], columns=FEATURE_NAMES)

    issues = []
    issue_scores = {}

    # 1. Hybrid CV Feature Trigger (Raised threshold to 22 to avoid natural texture false positives)
    noise_est = features.get("noise_estimate", 0.0)
    if noise_est > 22:
        sev = "medium" if noise_est < 35 else "high"
        issues.append({"type": "noise", "severity": sev, "confidence": 0.90, "evidence": {"noise_estimate": round(noise_est, 2)}})
        issue_scores["noise"] = 0.6 if sev == "medium" else 1.0

    # 2. ML Stage 1 and Stage 2 (Lowered threshold to 0.80 to catch Compression)
    prob_stage1 = clf_stage1.predict_proba(X)[0]
    degraded_prob = float(prob_stage1[list(le_stage1.classes_).index(1)])

    if degraded_prob > 0.80:
        pred_stage2 = clf_stage2.predict(X)[0]
        prob_stage2 = clf_stage2.predict_proba(X)[0]
        deg_type = le_stage2.inverse_transform([pred_stage2])[0]
        confidence = float(prob_stage2[pred_stage2])

        # 3. Severity based on CV features
        sev = "low"
        if deg_type in ["blur", "motion_blur"]:
            laplacian = features.get("laplacian_variance", 0)
            if laplacian < 20: sev = "high"
            elif laplacian < 100: sev = "medium"
        elif deg_type == "underexposure":
            brightness = features.get("mean_brightness", 128)
            if brightness < 40: sev = "high"
            elif brightness < 70: sev = "medium"
        elif deg_type == "overexposure":
            brightness = features.get("mean_brightness", 128)
            if brightness > 190: sev = "high"
            elif brightness > 160: sev = "medium"
        elif deg_type == "compression":
            sev = "medium"

        if sev != "low" or confidence > 0.85:
            evidence = {}
            if deg_type in ["blur", "motion_blur"]:
                evidence["laplacian_variance"] = round(features.get("laplacian_variance", 0), 2)
            elif deg_type in ["underexposure", "overexposure"]:
                evidence["mean_brightness"] = round(features.get("mean_brightness", 0), 2)
            elif deg_type == "noise":
                evidence["noise_estimate"] = round(features.get("noise_estimate", 0), 2)
            elif deg_type == "compression":
                evidence["white_clip_ratio"] = round(features.get("white_clip_ratio", 0), 4)

            issues.append({
                "type": deg_type,
                "severity": sev,
                "confidence": confidence_from_probability(confidence),
                "evidence": evidence
            })
            issue_scores[deg_type] = {"low": 0.3, "medium": 0.6, "high": 1.0}[sev]

    if not issues:
        degraded_prob = 0.0

    score = calculate_quality_score(issue_scores)
    label = quality_label_from_score(score)

    return {
        "quality_score": score,
        "quality_label": label,
        "clean_vs_degraded_prob": round(degraded_prob, 3),
        "issues": issues
    }
