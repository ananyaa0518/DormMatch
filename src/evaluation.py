"""Evaluation helpers for clustering and roommate recommendations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    recommended = recommended[:k]
    return len(set(recommended).intersection(relevant)) / k if k else 0.0


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    return len(set(recommended[:k]).intersection(relevant)) / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended: list, relevance: dict, k: int) -> float:
    dcg = sum(relevance.get(item, 0) / np.log2(rank + 2)
              for rank, item in enumerate(recommended[:k]))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(score / np.log2(rank + 2) for rank, score in enumerate(ideal))
    return float(dcg / idcg) if idcg else 0.0


def synthetic_compatibility_ground_truth(df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Build pairwise labels from a documented weighted lifestyle rule.

    This ground truth is intentionally generated for synthetic evaluation only:
    it combines sleep timing, study, cleanliness, noise, social behavior, and
    guest preference. It does not use CGPA, branch, gender, or model predictions.
    """
    n = len(df)
    matrix = np.zeros((n, n), dtype=float)
    relevance = {}
    for i in range(n):
        for j in range(i + 1, n):
            sleep_gap = abs(float(df.iloc[i]["sleep_time"]) - float(df.iloc[j]["sleep_time"]))
            sleep_gap = min(sleep_gap, 24 - sleep_gap)
            wake_gap = abs(float(df.iloc[i]["wake_time"]) - float(df.iloc[j]["wake_time"]))
            wake_gap = min(wake_gap, 24 - wake_gap)
            differences = [
                1 - min(sleep_gap / 12, 1),
                1 - abs(df.iloc[i]["study_hours"] - df.iloc[j]["study_hours"]) / 18,
                1 - abs(df.iloc[i]["cleanliness_level"] - df.iloc[j]["cleanliness_level"]) / 4,
                1 - abs(df.iloc[i]["noise_tolerance"] - df.iloc[j]["noise_tolerance"]) / 4,
                1 - abs(df.iloc[i]["social_activity"] - df.iloc[j]["social_activity"]) / 4,
            ]
            score = float(np.clip(np.mean(differences), 0, 1))
            matrix[i, j] = matrix[j, i] = score
    labels = {i: set(np.where(matrix[i] >= 0.72)[0].tolist()) for i in range(n)}
    relevance["threshold"] = 0.72
    relevance["feature_rule"] = "Mean of normalized sleep, study, cleanliness, noise, and social compatibility"
    return matrix, relevance


def recommendation_metrics(recommended_ids: dict, ground_truth: dict,
                           k: int = 5) -> pd.DataFrame:
    rows = []
    for student_id, recommendations in recommended_ids.items():
        relevant = ground_truth.get(student_id, set())
        relevance = {candidate: 1 for candidate in relevant}
        recs = recommendations[:k]
        rows.append({
            "student_id": student_id,
            "precision": precision_at_k(recs, relevant, k),
            "recall": recall_at_k(recs, relevant, k),
            "ndcg": ndcg_at_k(recs, relevance, k),
        })
    return pd.DataFrame(rows)


def catalog_coverage(recommendations: dict, catalog_size: int) -> float:
    recommended = set().union(*map(set, recommendations.values())) if recommendations else set()
    return len(recommended) / catalog_size if catalog_size else 0.0


def recommendation_diversity(recommended_ids: dict, cluster_assignments: pd.Series) -> float:
    overlap = []
    values = list(recommended_ids.values())
    for i, recs in enumerate(values):
        clusters = [cluster_assignments.get(r, -1) for r in recs]
        overlap.append(len(set(clusters)) / len(clusters) if clusters else 0)
    return float(np.mean(overlap))