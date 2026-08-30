"""Calibrate the trained CatBoost model using isotonic regression."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import GroupShuffleSplit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.model.features import ENGINEERED_FEATURES, compute_engineered_features

RAW_MODEL_PATH = Path("artifacts/models/best-next-correct.joblib")
CAL_MODEL_PATH = Path("artifacts/models/best-next-correct-calibrated.joblib")
DATASET_PATH = Path("artifacts/datasets/interactions.csv")
PLOT_PATH = Path("artifacts/benchmarks/calibration-curve.png")


def main():
    # 1. Load raw model artifact
    print("Loading model artifact...")
    artifact = joblib.load(RAW_MODEL_PATH)
    pipeline = artifact["model"]  # sklearn Pipeline: ColumnTransformer → CatBoostClassifier
    feature_cols = artifact["features"]
    target_col = artifact["target"]

    # 2. Load dataset and compute engineered features
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    df = compute_engineered_features(df)

    # Create target: next_correct (shift correct by -1 within each learner)
    df[target_col] = (
        df.groupby("learner_id")["correct"]
        .shift(-1)
        .fillna(0)
        .astype(int)
    )

    # 3. Split by learner_id: same 75/25 as the original training
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df["learner_id"]))
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    # 4. Carve 10% calibration set from training data (grouped by learner_id)
    cal_splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
    _, cal_idx = next(cal_splitter.split(train_df, groups=train_df["learner_id"]))
    cal_df = train_df.iloc[cal_idx]

    X_cal = cal_df[feature_cols]
    y_cal = cal_df[target_col]
    print(f"Calibration set: {len(cal_df)} rows from {cal_df['learner_id'].nunique()} learners")

    # 5. Calibrate using isotonic regression (prefit — the Pipeline is already trained)
    print("Calibrating with isotonic regression...")
    calibrated = CalibratedClassifierCV(pipeline, method="isotonic", cv="prefit")
    calibrated.fit(X_cal, y_cal)

    # 6. Plot reliability diagram: raw vs calibrated on the TEST set
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    prob_raw = pipeline.predict_proba(X_test)[:, 1]
    prob_cal = calibrated.predict_proba(X_test)[:, 1]

    frac_raw, mean_raw = calibration_curve(y_test, prob_raw, n_bins=10, strategy="uniform")
    frac_cal, mean_cal = calibration_curve(y_test, prob_cal, n_bins=10, strategy="uniform")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    ax1.plot(mean_raw, frac_raw, "s-", color="#e74c3c", label=f"Raw CatBoost")
    ax1.plot(mean_cal, frac_cal, "o-", color="#2ecc71", label=f"Isotonic Calibrated")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title("Reliability Diagram")
    ax1.legend(loc="lower right")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)

    # Prediction distribution
    ax2.hist(prob_raw, bins=50, alpha=0.5, color="#e74c3c", label="Raw", density=True)
    ax2.hist(prob_cal, bins=50, alpha=0.5, color="#2ecc71", label="Calibrated", density=True)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Density")
    ax2.set_title("Prediction Distribution")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"Saved calibration curve to {PLOT_PATH}")

    # 7. Compute calibration error metrics
    from sklearn.metrics import brier_score_loss
    brier_raw = brier_score_loss(y_test, prob_raw)
    brier_cal = brier_score_loss(y_test, prob_cal)
    print(f"Brier score — Raw: {brier_raw:.4f}, Calibrated: {brier_cal:.4f}")

    # 8. Save calibrated model
    cal_artifact = {
        **artifact,
        "model": calibrated,
        "model_name": f"{artifact['model_name']}_calibrated",
        "brier_raw": brier_raw,
        "brier_calibrated": brier_cal,
    }
    joblib.dump(cal_artifact, CAL_MODEL_PATH)
    print(f"Saved calibrated model to {CAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
