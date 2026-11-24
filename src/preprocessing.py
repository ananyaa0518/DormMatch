"""Preprocessing helpers shared by recommendations and the Streamlit app."""

from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


LIFESTYLE_FEATURES = [
    "sleep_time_sin", "sleep_time_cos", "wake_time_sin", "wake_time_cos",
    "sleep_duration", "cleanliness_level", "noise_tolerance", "study_hours",
    "social_activity", "introvert_extrovert", "sharing_preference",
    "weekend_activity", "food_habit", "smoking_preference", "ac_preference",
    "preferred_room_type", "preferred_environment", "roommate_interaction",
]
CONTINUOUS_FEATURES = [
    "sleep_time_sin", "sleep_time_cos", "wake_time_sin", "wake_time_cos",
    "sleep_duration", "cleanliness_level", "noise_tolerance", "study_hours",
    "social_activity", "introvert_extrovert", "sharing_preference",
    "roommate_interaction",
]


def circular_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sleep_time"] = pd.to_numeric(out["sleep_time"], errors="coerce").clip(0, 24) % 24
    out["wake_time"] = pd.to_numeric(out["wake_time"], errors="coerce").clip(0, 24) % 24
    out["sleep_time_sin"] = np.sin(2 * np.pi * out["sleep_time"] / 24)
    out["sleep_time_cos"] = np.cos(2 * np.pi * out["sleep_time"] / 24)
    out["wake_time_sin"] = np.sin(2 * np.pi * out["wake_time"] / 24)
    out["wake_time_cos"] = np.cos(2 * np.pi * out["wake_time"] / 24)
    return out


def fit_mixed_preprocessor(feature_columns: list[str], engineered: pd.DataFrame) -> tuple[StandardScaler, list[str]]:
    data = circular_time_features(engineered)
    for column in ["sharing_preference", "roommate_interaction"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    encoded = pd.get_dummies(data[LIFESTYLE_FEATURES], drop_first=False, dtype=int)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    scaler = StandardScaler()
    available = [c for c in CONTINUOUS_FEATURES if c in encoded.columns]
    scaler.fit(encoded[available])
    return scaler, available


def transform_profile(profile: dict, feature_columns: list[str], scaler: StandardScaler,
                     continuous_columns: list[str]) -> np.ndarray:
    row = pd.DataFrame([profile])
    row = circular_time_features(row)
    for column in ["sharing_preference", "roommate_interaction"]:
        row[column] = pd.to_numeric(row[column], errors="coerce")
    encoded = pd.get_dummies(row[LIFESTYLE_FEATURES], drop_first=False, dtype=int)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    encoded[continuous_columns] = scaler.transform(encoded[continuous_columns])
    return encoded.to_numpy(dtype=float)


def save_preprocessing(path: str, scaler: StandardScaler, continuous_columns: list[str],
                       feature_columns: list[str]) -> None:
    obj = {
        "scaler": scaler,
        "continuous_columns": continuous_columns,
        "feature_columns": feature_columns,
    }
    joblib.dump(obj, path)