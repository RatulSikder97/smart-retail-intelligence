"""Phase 5 (RQ4) — shops that behave anomalously versus their cluster peers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from smart_retail.config import Config
from smart_retail.preprocessing import schema
from smart_retail.preprocessing.features import TARGET_COLUMN, feature_matrix


@dataclass
class AnomalyResult:
    scores: pd.DataFrame          # per shop: cluster, iforest/lof percentile ranks, combined
    top_anomalies: pd.DataFrame   # top decile with reporting context
    n_skipped_small_clusters: int


def _cluster_scores(X: np.ndarray, config: Config) -> tuple[np.ndarray, np.ndarray]:
    # returns percentile ranks 0-1 per method, higher = more anomalous
    X = StandardScaler().fit_transform(X)

    iforest = IsolationForest(
        n_estimators=200, random_state=config.random_state, n_jobs=config.n_jobs
    )
    iforest.fit(X)
    # score_samples: lower = more anomalous → negate before ranking
    iforest_rank = pd.Series(-iforest.score_samples(X)).rank(pct=True).to_numpy()

    n_neighbors = min(20, len(X) - 1)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, n_jobs=config.n_jobs)
    lof.fit(X)
    lof_rank = pd.Series(-lof.negative_outlier_factor_).rank(pct=True).to_numpy()

    return iforest_rank, lof_rank


def run_anomaly(
    features: pd.DataFrame, cluster_labels: pd.Series, config: Config
) -> AnomalyResult:
    X_all = feature_matrix(features).to_numpy()

    scores = pd.DataFrame(
        {
            schema.SHOP_ID: features[schema.SHOP_ID].values,
            "cluster": cluster_labels.values,
            "iforest_rank": np.nan,
            "lof_rank": np.nan,
        },
        index=features.index,
    )

    n_skipped = 0
    for cluster_id, idx in features.groupby(cluster_labels.values).groups.items():
        if len(idx) < config.anomaly_min_cluster_size:
            n_skipped += 1
            continue
        positions = features.index.get_indexer(idx)
        iforest_rank, lof_rank = _cluster_scores(X_all[positions], config)
        scores.loc[idx, "iforest_rank"] = iforest_rank
        scores.loc[idx, "lof_rank"] = lof_rank

    scores["combined_score"] = scores[["iforest_rank", "lof_rank"]].mean(axis=1)
    scored = scores.dropna(subset=["combined_score"]).sort_values(
        "combined_score", ascending=False
    )

    n_top = max(1, int(round(len(scored) * config.anomaly_top_decile)))
    context_cols = [
        c
        for c in (
            schema.SHOP_ID, "division", "zila", "shop_type", "shop_importance",
            "density_1km", "dist_to_district_centre_km", TARGET_COLUMN,
            "latitude", "longitude",
        )
        if c in features.columns
    ]
    top_anomalies = (
        scored.head(n_top)[[schema.SHOP_ID, "cluster", "combined_score"]]
        .merge(features[context_cols], on=schema.SHOP_ID, how="left")
        .reset_index(drop=True)
    )

    return AnomalyResult(
        scores=scores.reset_index(drop=True),
        top_anomalies=top_anomalies,
        n_skipped_small_clusters=n_skipped,
    )
