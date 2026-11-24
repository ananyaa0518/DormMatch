"""Feature engineering helpers for DormMatch."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


LIFESTYLE_FEATURES = [
    "sleep_time_sin",
    "sleep_time_cos",
    "wake_time_sin",
    "wake_time_cos",
    "sleep_duration",
    "cleanliness_level",
    "noise_tolerance",
    "study_hours",
    "social_activity",
    "introvert_extrovert",
    "sharing_preference",
    "weekend_activity",
    "food_habit",
    "smoking_preference",
    "ac_preference",
    "preferred_room_type",
    "preferred_environment",
    "roommate_interaction",
]

ORDINAL_FEATURES = ["sharing_preference", "roommate_interaction"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add circular time features and exploratory lifestyle categories."""
    engineered = df.copy()
    engineered["sleep_time"] = engineered["sleep_time"] % 24
    engineered["wake_time"] = engineered["wake_time"] % 24

    engineered["sleep_time_sin"] = np.sin(2 * np.pi * engineered["sleep_time"] / 24)
    engineered["sleep_time_cos"] = np.cos(2 * np.pi * engineered["sleep_time"] / 24)
    engineered["wake_time_sin"] = np.sin(2 * np.pi * engineered["wake_time"] / 24)
    engineered["wake_time_cos"] = np.cos(2 * np.pi * engineered["wake_time"] / 24)

    engineered["study_intensity"] = pd.cut(
        engineered["study_hours"],
        bins=[-np.inf, 2, 5, 8, np.inf],
        labels=["Low", "Moderate", "High", "Very High"],
    )
    engineered["sleep_category"] = pd.cut(
        engineered["sleep_duration"],
        bins=[0, 5, 7, 9, np.inf],
        labels=["Short", "Moderate", "Ideal", "Long"],
        include_lowest=True,
    )
    engineered["social_category"] = pd.cut(
        engineered["social_activity"],
        bins=[0, 2, 3, 5],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )

    for column in ORDINAL_FEATURES:
        engineered[column] = pd.to_numeric(engineered[column], errors="coerce")

    return engineered


def encode_lifestyle_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Encode and scale lifestyle features in both recommended clustering forms."""
    df_encoded = pd.get_dummies(
        df[LIFESTYLE_FEATURES],
        drop_first=False,
        dtype=int,
    )

    all_scaler = StandardScaler()
    X_scaled_all = pd.DataFrame(
        all_scaler.fit_transform(df_encoded),
        columns=df_encoded.columns,
        index=df_encoded.index,
    )

    continuous_features = [
        "sleep_time_sin",
        "sleep_time_cos",
        "wake_time_sin",
        "wake_time_cos",
        "sleep_duration",
        "cleanliness_level",
        "noise_tolerance",
        "study_hours",
        "social_activity",
        "introvert_extrovert",
        "sharing_preference",
        "roommate_interaction",
    ]
    X_scaled_mixed = df_encoded.copy()
    mixed_scaler = StandardScaler()
    X_scaled_mixed[continuous_features] = mixed_scaler.fit_transform(
        X_scaled_mixed[continuous_features]
    )

    return df_encoded, X_scaled_all, X_scaled_mixed, mixed_scaler