"""Roommate recommendation models and compatibility explanations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity


WEIGHTS = {
    "sleep": 0.20,
    "study": 0.15,
    "cleanliness": 0.15,
    "noise": 0.12,
    "social": 0.12,
    "guest": 0.07,
    "music": 0.07,
    "preferences": 0.12,
}


def circular_difference(a: float, b: float, period: float = 24.0) -> float:
    a = float(a) % period
    b = float(b) % period
    return abs((a - b + period / 2) % period - period / 2)


def scaled_similarity(a: float, b: float, low: float, high: float) -> float:
    return float(1 - min(abs(float(a) - float(b)) / (high - low), 1))


def knn_recommend(feature_matrix: np.ndarray, top_k: int = 5,
                  query_index: int | None = None,
                  query_vector: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    model = NearestNeighbors(n_neighbors=min(top_k + 1, len(feature_matrix)), metric="euclidean")
    model.fit(feature_matrix)
    vector = feature_matrix[query_index:query_index + 1] if query_index is not None else query_vector
    distances, indices = model.kneighbors(vector)
    if query_index is not None:
        keep = indices[0] != query_index
        indices = indices[:, keep][:, :top_k]
        distances = distances[:, keep][:, :top_k]
    else:
        indices = indices[:, :top_k]
        distances = distances[:, :top_k]
    return indices[0], distances[0]


def cosine_recommend(feature_matrix: np.ndarray, top_k: int = 5,
                     query_index: int | None = None,
                     query_vector: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    vector = feature_matrix[query_index:query_index + 1] if query_index is not None else query_vector
    scores = cosine_similarity(vector, feature_matrix)[0]
    if query_index is not None:
        scores[query_index] = -np.inf
    indices = np.argsort(-scores)[:top_k]
    return indices, scores[indices]


GUEST_MAP = {"rarely": 1, "sometimes": 2, "often": 3, "very often": 4}
MUSIC_MAP = {"never": 1, "rarely": 2, "sometimes": 3, "often": 4}


def category_score(a: object, b: object) -> float:
    return 1.0 if str(a).strip().lower() == str(b).strip().lower() else 0.0


def weighted_compatibility(query: pd.Series, candidate: pd.Series,
                           weights: dict[str, float] | None = None) -> dict:
    weights = WEIGHTS if weights is None else weights
    sleep_distance = (circular_difference(query["sleep_time"], candidate["sleep_time"]) +
                      circular_difference(query["wake_time"], candidate["wake_time"])) / 2
    sleep = 1 - min(sleep_distance / 12, 1)
    study = scaled_similarity(query["study_hours"], candidate["study_hours"], 0, 18)
    cleanliness = scaled_similarity(query["cleanliness_level"], candidate["cleanliness_level"], 1, 5)
    noise = scaled_similarity(query["noise_tolerance"], candidate["noise_tolerance"], 1, 5)
    social = scaled_similarity(query["social_activity"], candidate["social_activity"], 1, 5)

    guest_a = GUEST_MAP.get(str(query.get("guest_frequency", "rarely")).strip().lower(), 1)
    guest_b = GUEST_MAP.get(str(candidate.get("guest_frequency", "rarely")).strip().lower(), 1)
    guest = 1 - abs(guest_a - guest_b) / 3
    music_a = MUSIC_MAP.get(str(query.get("music_frequency", "never")).strip().lower(), 1)
    music_b = MUSIC_MAP.get(str(candidate.get("music_frequency", "never")).strip().lower(), 1)
    music = 1 - abs(music_a - music_b) / 3

    preference_fields = ["food_habit", "preferred_room_type", "ac_preference", "preferred_environment"]
    preference_scores = [category_score(query.get(field), candidate.get(field)) for field in preference_fields]
    preferences = float(np.mean(preference_scores))
    components = {
        "sleep": sleep, "study": study, "cleanliness": cleanliness, "noise": noise,
        "social": social, "guest": guest, "music": music, "preferences": preferences,
    }
    overall = 100 * sum(components[key] * weights[key] for key in WEIGHTS)
    return {"overall": overall, **{key: 100 * value for key, value in components.items()},
            "sleep_difference_hours": sleep_distance}


def weighted_recommend(query: pd.Series, candidates: pd.DataFrame, top_k: int = 5,
                       constraints: dict | None = None) -> pd.DataFrame:
    constraints = constraints or {}
    eligible = candidates.copy()
    constraint_map = {
        "room_type": "preferred_room_type",
        "ac_preference": "ac_preference",
        "food_preference": "food_habit",
        "preferred_environment": "preferred_environment",
    }
    for key, value in constraints.items():
        column = constraint_map.get(key, key)
        if column in eligible.columns and value not in (None, "", "Any"):
            eligible = eligible[eligible[column].astype(str).str.lower().str.strip()
                              == str(value).lower().strip()]

    rows = []
    for _, candidate in eligible.iterrows():
        if str(candidate["student_id"]) == str(query["student_id"]):
            continue
        result = weighted_compatibility(query, candidate)
        result["student_id"] = candidate["student_id"]
        result["cluster"] = candidate.get("cluster")
        rows.append(result)
    return pd.DataFrame(rows).sort_values("overall", ascending=False).head(top_k).reset_index(drop=True)