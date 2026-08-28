from __future__ import annotations

import os
import sys
import cv2
import numpy as np
import pandas as pd
import joblib
import time
from functools import lru_cache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vision.features import extract_features

ARTIFACT_DIR = os.path.join(ROOT, "ml", "artifacts")

clf_stage1 = joblib.load(
    os.path.join(ARTIFACT_DIR, "stage1_clean_vs_degraded.joblib")
)
le_stage1 = joblib.load(
    os.path.join(ARTIFACT_DIR, "stage1_label_encoder.joblib")
)

clf_stage2 = joblib.load(
    os.path.join(ARTIFACT_DIR, "stage2_degradation_type.joblib")
)
le_stage2 = joblib.load(
    os.path.join(ARTIFACT_DIR, "stage2_label_encoder.joblib")
)

CSV_PATH = os.path.join(
    ROOT,
    "dataset/metadata/feature_dataset.csv",
)

df_cols = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()

EXCLUDED_COLUMNS = [
    "source_id",
    "split",
    "variant_id",
    "degradation",
    "severity",
    "severity_name",
    "is_degraded",
]

FEATURE_NAMES = [
    c for c in df_cols
    if c not in EXCLUDED_COLUMNS
]


def _feature_frame(features):
    return pd.DataFrame(
        [[float(features.get(name, 0.0)) for name in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
    )


def _cv_candidates(features):
    """
    Independent CV evidence.

    These rules are deliberately conservative. They are used to
    prevent Stage 1's RF from hiding obvious degradations.
    """

    candidates = []

    brightness = features.get("mean_brightness", 128.0)
    dark_ratio = features.get("dark_ratio", 0.0)
    bright_ratio = features.get("bright_ratio", 0.0)
    black_clip = features.get("black_clip_ratio", 0.0)
    white_clip = features.get("white_clip_ratio", 0.0)

    laplacian = features.get("laplacian_variance", 0.0)
    noise = features.get("noise_estimate", 0.0)

    # Underexposure
    if (
        brightness < 70
        or dark_ratio > 0.35
        or black_clip > 0.03
    ):
        candidates.append("underexposure")

    # Overexposure
    if (
        brightness > 190
        or bright_ratio > 0.35
        or white_clip > 0.03
    ):
        candidates.append("overexposure")

    # Strong noise
    if noise > 18:
        candidates.append("noise")

    # Blur.
    # Keep this threshold lower than the old Stage-1 gate.
    if laplacian < 100:
        candidates.append("blur")

    return candidates


def _severity(issue_type, features):
    brightness = features.get("mean_brightness", 128.0)
    laplacian = features.get("laplacian_variance", 0.0)
    noise = features.get("noise_estimate", 0.0)

    if issue_type in ("blur", "motion_blur"):
        if laplacian < 20:
            return "high"
        if laplacian < 100:
            return "medium"
        return "low"

    if issue_type == "underexposure":
        if brightness < 40:
            return "high"
        if brightness < 70:
            return "medium"
        return "low"

    if issue_type == "overexposure":
        if brightness > 220:
            return "high"
        if brightness > 190:
            return "medium"
        return "low"

    if issue_type == "noise":
        if noise >= 35:
            return "high"
        if noise >= 18:
            return "medium"
        return "low"

    if issue_type == "compression":
        return "medium"

    return "low"


def _evidence(issue_type, features):
    if issue_type in ("blur", "motion_blur"):
        return {
            "laplacian_variance": round(
                features.get("laplacian_variance", 0), 2
            )
        }

    if issue_type in ("underexposure", "overexposure"):
        return {
            "mean_brightness": round(
                features.get("mean_brightness", 0), 2
            )
        }

    if issue_type == "noise":
        return {
            "noise_estimate": round(
                features.get("noise_estimate", 0), 2
            )
        }

    if issue_type == "compression":
        return {
            "entropy": round(
                features.get("entropy", 0), 3
            ),
            "laplacian_variance": round(
                features.get("laplacian_variance", 0), 2
            ),
        }

    return {}


def _add_issue(issues, issue_scores, issue_type, confidence, features):
    severity = _severity(issue_type, features)

    # Do not report weak RF predictions unless there is independent
    # CV evidence for the same class.
    if severity == "low" and confidence < 0.85:
        return

    issues.append(
        {
            "type": issue_type,
            "severity": severity,
            "confidence": round(float(confidence), 3),
            "evidence": _evidence(issue_type, features),
        }
    )

    issue_scores[issue_type] = {
        "low": 0.3,
        "medium": 0.6,
        "high": 1.0,
    }[severity]


@lru_cache(maxsize=128)
def cached_analyze(image_path: str):
    return analyze_image(image_path)


def analyze_image(image_path):
    start_time = time.time()

    image = cv2.imread(image_path)

    if image is None:
        raise RuntimeError(
            f"Could not load image: {image_path}"
        )

    features = extract_features(image)
    X = _feature_frame(features)

    issues = []
    issue_scores = {}

    # ---------------------------------------------------------
    # Stage 1
    # ---------------------------------------------------------

    prob_stage1 = clf_stage1.predict_proba(X)[0]

    stage1_class_1 = list(le_stage1.classes_).index(1)
    rf_degraded_prob = float(prob_stage1[stage1_class_1])

    # ---------------------------------------------------------
    # Independent CV gate
    #
    # The old code required RF probability > 0.80.
    # That caused many obvious generated degradations to be
    # treated as clean and prevented Stage 2 from running.
    # ---------------------------------------------------------

    cv_candidates = _cv_candidates(features)

    obvious_cv_degradation = bool(cv_candidates)

    should_run_stage2 = (
        rf_degraded_prob >= 0.45
        or obvious_cv_degradation
    )

    if should_run_stage2:

        pred_stage2 = int(clf_stage2.predict(X)[0])
        prob_stage2 = clf_stage2.predict_proba(X)[0]

        deg_type = le_stage2.inverse_transform(
            [pred_stage2]
        )[0]

        confidence = float(
            prob_stage2[pred_stage2]
        )

        # -----------------------------------------------------
        # Use independent CV evidence to correct obvious cases.
        # -----------------------------------------------------

        if cv_candidates:

            # Exposure has strong direct evidence.
            if "underexposure" in cv_candidates:
                if features.get("mean_brightness", 128) < 70:
                    deg_type = "underexposure"

            elif "overexposure" in cv_candidates:
                if (
                    features.get("mean_brightness", 128) > 190
                    or features.get("bright_ratio", 0) > 0.35
                ):
                    deg_type = "overexposure"

            # Noise has a direct measurable signal.
            elif "noise" in cv_candidates:
                if features.get("noise_estimate", 0) > 18:
                    deg_type = "noise"

            # If the RF thinks blur/motion blur, keep its distinction.
            elif deg_type in ("blur", "motion_blur"):
                pass

            # If RF predicts a clearly unrelated class but image has
            # strong blur evidence, use blur as the safer diagnosis.
            elif "blur" in cv_candidates:
                if features.get("laplacian_variance", 0) < 100:
                    deg_type = "blur"

        # -----------------------------------------------------
        # Motion blur vs normal blur
        #
        # The classifier's motion-blur distinction is retained when
        # confidence is strong. Otherwise normal blur is safer.
        # -----------------------------------------------------

        if deg_type == "motion_blur":
            if confidence < 0.60:
                deg_type = "blur"

        # -----------------------------------------------------
        # Add classification result.
        # -----------------------------------------------------

        _add_issue(
            issues,
            issue_scores,
            deg_type,
            confidence,
            features,
        )

    # ---------------------------------------------------------
    # Independent noise detector.
    #
    # This is intentionally separate from Stage 1/2 so noise
    # cannot disappear because of an incorrect RF gate.
    # ---------------------------------------------------------

    noise_est = features.get("noise_estimate", 0.0)

    if noise_est > 18:

        existing_noise = any(
            issue["type"] == "noise"
            for issue in issues
        )

        if not existing_noise:

            if noise_est >= 35:
                severity = "high"
            elif noise_est >= 22:
                severity = "medium"
            else:
                severity = "low"

            issues.append(
                {
                    "type": "noise",
                    "severity": severity,
                    "confidence": round(
                        min(0.99, 0.70 + noise_est / 100),
                        3,
                    ),
                    "evidence": {
                        "noise_estimate": round(
                            noise_est, 2
                        )
                    },
                }
            )

            issue_scores["noise"] = {
                "low": 0.3,
                "medium": 0.6,
                "high": 1.0,
            }[severity]

    # ---------------------------------------------------------
    # Independent exposure protection.
    # ---------------------------------------------------------

    brightness = features.get(
        "mean_brightness", 128.0
    )

    if brightness < 70:

        if not any(
            issue["type"] == "underexposure"
            for issue in issues
        ):
            _add_issue(
                issues,
                issue_scores,
                "underexposure",
                0.95,
                features,
            )

    if (
        brightness > 190
        or features.get("bright_ratio", 0) > 0.35
    ):

        if not any(
            issue["type"] == "overexposure"
            for issue in issues
        ):
            _add_issue(
                issues,
                issue_scores,
                "overexposure",
                0.95,
                features,
            )

    # ---------------------------------------------------------
    # Final probability.
    #
    # Keep the RF probability when useful, but ensure obvious
    # CV degradation cannot be reported as exactly 0.
    # ---------------------------------------------------------

    degraded_prob = rf_degraded_prob

    if issues:
        degraded_prob = max(
            degraded_prob,
            0.80,
        )

    score = calculate_quality_score(
        issue_scores
    )

    label = quality_label_from_score(
        score
    )

    elapsed = time.time() - start_time

    return {
        "quality_score": score,
        "quality_label": label,
        "clean_vs_degraded_prob": round(
            float(degraded_prob), 3
        ),
        "issues": issues,
        "inference_time_ms": round(
            elapsed * 1000, 2
        ),
    }


def calculate_quality_score(issue_scores):
    if not issue_scores:
        return 100.0

    weights = {
        "blur": 25.0,
        "motion_blur": 25.0,
        "underexposure": 20.0,
        "overexposure": 20.0,
        "noise": 20.0,
        "compression": 20.0,
    }

    total_penalty = 0.0

    for issue_type, severity_score in issue_scores.items():

        weight = weights.get(
            issue_type,
            15.0,
        )

        severity_score = max(
            0.0,
            min(1.0, severity_score),
        )

        total_penalty += (
            weight * severity_score
        )

    return round(
        max(
            0.0,
            min(
                100.0,
                100.0 - total_penalty,
            ),
        ),
        2,
    )


def quality_label_from_score(score):
    if score >= 80:
        return "ACCEPTABLE"

    if score >= 50:
        return "DEGRADED"

    return "POTENTIALLY_DEFECTIVE"
