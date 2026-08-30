"""Deep Behavioral Feature Engineering for Adaptive Learning System."""
from __future__ import annotations

import pandas as pd
import numpy as np


BASE_FEATURES = [
    "difficulty_score", "knowledge_before", "fatigue_before", "total_response_time",
    "reading_time", "time_after_last_interaction", "attempts", "skip", "option_changes",
    "mouse_distance", "mouse_speed", "hover_time", "typing_speed", "backspaces",
    "delete_frequency", "pause_duration", "question_number", "session_duration", "tab_switches",
]

ENGINEERED_FEATURES = BASE_FEATURES + [
    "rolling_accuracy_3",
    "response_time_drift",
    "friction_index",
    "engagement_drop",
    "accuracy_x_difficulty",
]


def compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes leakage-safe derived behavioral signals per learner."""
    df = df.copy()

    # Sort to preserve correct chronological sequence per learner
    if "event_index" in df.columns:
        df = df.sort_values(["learner_id", "event_index"])

    # 1. Rolling accuracy over last 3 attempts
    if "correct" in df.columns:
        df["rolling_accuracy_3"] = (
            df.groupby("learner_id")["correct"]
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
            .fillna(0.5)
        )
    else:
        df["rolling_accuracy_3"] = 0.5

    # 2. Response time drift (ratio of current RT to learner cumulative mean RT)
    cum_rt_mean = (
        df.groupby("learner_id")["total_response_time"]
        .transform(lambda s: s.shift(1).expanding().mean())
        .fillna(df["total_response_time"])
    )
    df["response_time_drift"] = (df["total_response_time"] / (cum_rt_mean + 1e-5)).clip(0, 5)

    # 3. Friction index: composite score of UI hesitation & corrections
    df["friction_index"] = (
        (df["backspaces"] + df["delete_frequency"] + df["pause_duration"])
        / (df["total_response_time"] + 1.0)
    ).round(6)

    # 4. Engagement drop indicator
    df["engagement_drop"] = (
        (df["tab_switches"] > 0) | (df["time_after_last_interaction"] > 8.0) | (df["skip"] == 1)
    ).astype(float)

    # 5. Accuracy interaction term (if correctness is available, otherwise estimated)
    if "correct" in df.columns:
        df["accuracy_x_difficulty"] = df["correct"].astype(float) * df["difficulty_score"]
    else:
        df["accuracy_x_difficulty"] = df["knowledge_before"] * df["difficulty_score"]

    return df
