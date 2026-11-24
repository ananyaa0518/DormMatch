"""Lifestyle clustering helpers for DormMatch."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


RANDOM_STATE = 42


def evaluate_kmeans(
    feature_matrix: pd.DataFrame,
    k_values: range = range(2, 11),
) -> tuple[pd.DataFrame, dict[int, KMeans]]:
    """Fit K-Means models and calculate clustering metrics for each K."""
    X = feature_matrix.to_numpy()
    metrics = []
    models: dict[int, KMeans] = {}

    for k in k_values:
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20,
        )
        labels = model.fit_predict(X)
        models[k] = model
        metrics.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(X, labels),
                "davies_bouldin": davies_bouldin_score(X, labels),
                "calinski_harabasz": calinski_harabasz_score(X, labels),
            }
        )

    return pd.DataFrame(metrics), models


def choose_k(metrics: pd.DataFrame) -> int:
    """Choose K by the highest silhouette score."""
    return int(metrics.loc[metrics["silhouette"].idxmax(), "k"])


def fit_pca(feature_matrix: pd.DataFrame) -> tuple[PCA, np.ndarray]:
    """Fit a two-component PCA representation for visualization."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coordinates = pca.fit_transform(feature_matrix)
    return pca, coordinates


def build_cluster_profiles(
    cleaned_df: pd.DataFrame,
    assignments: pd.DataFrame,
    raw_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create mean lifestyle profiles with IDs aligned to the assignments."""
    profile_df = cleaned_df.copy()
    if "cluster" not in profile_df.columns:
        profile_df = profile_df.merge(
            assignments[["student_id", "cluster"]],
            on="student_id",
            how="inner",
            validate="one_to_one",
        )

    if raw_df is not None and {"guest_frequency", "music_frequency"}.issubset(
        raw_df.columns
    ):
        raw_profiles = (
            raw_df[["student_id", "guest_frequency", "music_frequency"]]
            .drop_duplicates(subset="student_id", keep="first")
        )
        for column in ["guest_frequency", "music_frequency"]:
            raw_profiles[column] = (
                raw_profiles[column]
                .astype("string")
                .str.strip()
                .str.lower()
                .map(
                    {
                        "rarely": 1,
                        "sometimes": 2,
                        "often": 3,
                        "very often": 4,
                    }
                )
            )
        profile_df = profile_df.merge(
            raw_profiles,
            on="student_id",
            how="left",
            validate="one_to_one",
        )

    numeric_columns = [
        "sleep_time",
        "wake_time",
        "sleep_duration",
        "study_hours",
        "cleanliness_level",
        "noise_tolerance",
        "social_activity",
        "introvert_extrovert",
        "guest_frequency",
        "music_frequency",
    ]
    available_columns = [c for c in numeric_columns if c in profile_df.columns]
    profiles = (
        profile_df.groupby("cluster", as_index=False)[available_columns]
        .mean()
        .sort_values("cluster")
    )
    cluster_sizes = profile_df["cluster"].value_counts().sort_index()
    profiles["cluster_size"] = profiles["cluster"].map(cluster_sizes)
    return profiles


def save_clustering_artifacts(
    models_dir: str,
    assignments: pd.DataFrame,
    profiles: pd.DataFrame,
    metrics: pd.DataFrame,
    selected_model: KMeans,
    pca: PCA,
    feature_columns: list[str],
) -> None:
    """Persist clustering assignments, profiles, metrics, and trained objects."""
    directory = Path(models_dir)
    directory.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(directory / "cluster_assignments.csv", index=False)
    profiles.to_csv(directory / "cluster_profiles.csv", index=False)
    metrics.to_csv(directory / "clustering_metrics.csv", index=False)
    joblib.dump(selected_model, directory / "kmeans_model.joblib")
    joblib.dump(pca, directory / "pca_model.joblib")
    joblib.dump(feature_columns, directory / "clustering_feature_columns.joblib")