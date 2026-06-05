"""Phase 2 (RQ1) — shop archetypes via K-Means, Hierarchical, and DBSCAN."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from smart_retail.config import Config
from smart_retail.preprocessing.features import feature_matrix

_SILHOUETTE_SAMPLE = 10_000


@dataclass
class ClusteringResult:
    labels: pd.Series          # cluster id per shop, aligned to features index
    best_k: int
    metrics: dict = field(default_factory=dict)
    profiles: pd.DataFrame = None
    scaler: StandardScaler = None
    model: KMeans = None


def _score(X: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> dict:
    if len(set(labels)) < 2:
        return {"silhouette": float("nan"), "davies_bouldin": float("nan")}
    idx = np.arange(len(X))
    if len(idx) > _SILHOUETTE_SAMPLE:
        idx = rng.choice(idx, _SILHOUETTE_SAMPLE, replace=False)
        # silhouette needs ≥2 clusters in the subsample too
        if len(set(labels[idx])) < 2:
            return {"silhouette": float("nan"), "davies_bouldin": float("nan")}
    return {
        "silhouette": float(silhouette_score(X[idx], labels[idx])),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }


def _profile_clusters(features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    df = features.assign(cluster=labels.values)
    numeric_keys = [
        c
        for c in (
            "density_500m", "density_1km", "density_5km",
            "dist_to_district_centre_km", "n_products_targeted", "total_target",
            "latitude", "longitude",
        )
        if c in df.columns
    ]
    profile = df.groupby("cluster")[numeric_keys].median()
    profile["n_shops"] = df.groupby("cluster").size()
    profile["share"] = (profile["n_shops"] / len(df)).round(4)
    for cat in ("division", "shop_type", "shop_importance"):
        if cat in df.columns:
            profile[f"top_{cat}"] = (
                df.groupby("cluster")[cat].agg(lambda s: s.mode().iat[0])
            )
    return profile.reset_index()


def bootstrap_stability(
    X: np.ndarray, reference: np.ndarray, k: int, config: Config
) -> dict:
    # refit K-Means on bootstrap resamples; ARI against the reference partition
    # measures how much the taxonomy depends on the particular sample drawn
    rng = np.random.default_rng(config.random_state)
    n = int(min(len(X), config.stability_sample_size))
    idx = rng.choice(len(X), n, replace=False)
    Xs, ref = X[idx], reference[idx]

    aris: list[float] = []
    cluster_ids = np.unique(ref)
    jaccard_runs: dict[int, list[float]] = {int(c): [] for c in cluster_ids}
    for b in range(config.stability_bootstraps):
        boot = rng.choice(n, n, replace=True)
        km = KMeans(n_clusters=k, n_init=3, random_state=config.random_state + b)
        km.fit(Xs[boot])
        pred = km.predict(Xs)
        aris.append(float(adjusted_rand_score(ref, pred)))
        # per-archetype recovery: best Jaccard overlap with any bootstrap cluster
        for c in cluster_ids:
            ref_mask = ref == c
            best = 0.0
            for p in np.unique(pred):
                pred_mask = pred == p
                inter = int(np.logical_and(ref_mask, pred_mask).sum())
                union = int(np.logical_or(ref_mask, pred_mask).sum())
                if union:
                    best = max(best, inter / union)
            jaccard_runs[int(c)].append(best)

    return {
        "n_bootstraps": config.stability_bootstraps,
        "sample_size": n,
        "ari_mean": round(float(np.mean(aris)), 4),
        "ari_std": round(float(np.std(aris)), 4),
        "ari_min": round(float(np.min(aris)), 4),
        "per_cluster_jaccard": {
            str(c): round(float(np.mean(v)), 4) for c, v in jaccard_runs.items()
        },
    }


def run_clustering(features: pd.DataFrame, config: Config) -> ClusteringResult:
    X_df = feature_matrix(features)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df)
    rng = np.random.default_rng(config.random_state)

    k_lo, k_hi = config.kmeans_k_range
    kmeans_metrics: dict[int, dict] = {}
    fitted: dict[int, KMeans] = {}
    for k in range(k_lo, k_hi + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=config.random_state)
        labels = km.fit_predict(X)
        kmeans_metrics[k] = _score(X, labels, rng)
        fitted[k] = km

    best_k = max(kmeans_metrics, key=lambda k: kmeans_metrics[k]["silhouette"])
    best_model = fitted[best_k]
    labels = pd.Series(best_model.predict(X), index=features.index, name="cluster")

    # Agglomerative/DBSCAN cross-check on a subsample — O(n²) at full scale
    sub = rng.choice(
        len(X), min(config.hierarchical_sample_size, len(X)), replace=False
    )
    agg_labels = AgglomerativeClustering(n_clusters=best_k).fit_predict(X[sub])
    dbscan_labels = DBSCAN(
        eps=config.dbscan_eps, min_samples=config.dbscan_min_samples
    ).fit_predict(X[sub])
    n_dbscan_clusters = int(len(set(dbscan_labels) - {-1}))

    metrics = {
        "kmeans": {str(k): m for k, m in kmeans_metrics.items()},
        "agglomerative": _score(X[sub], agg_labels, rng),
        "dbscan": {
            **_score(X[sub][dbscan_labels != -1], dbscan_labels[dbscan_labels != -1], rng),
            "n_clusters_found": n_dbscan_clusters,
            "noise_share": float((dbscan_labels == -1).mean()),
        },
        "best_k": best_k,
        "chosen_algorithm": "kmeans",
        "stability": bootstrap_stability(X, labels.to_numpy(), best_k, config),
    }

    return ClusteringResult(
        labels=labels,
        best_k=best_k,
        metrics=metrics,
        profiles=_profile_clusters(features, labels),
        scaler=scaler,
        model=best_model,
    )
