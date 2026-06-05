import numpy as np
import pytest

from smart_retail.config import Config
from smart_retail.features.engineering import run_preprocessing
from smart_retail.models.anomaly import run_anomaly
from smart_retail.models.clustering import run_clustering


@pytest.fixture(scope="module")
def anomalies(shops_sample, targets_sample, config):
    features, _ = run_preprocessing(shops_sample, targets_sample, config)
    cfg = Config(kmeans_k_range=(4, 4), hierarchical_sample_size=2_000)
    clusters = run_clustering(features, cfg)
    return features, run_anomaly(features, clusters.labels, cfg)


def test_every_shop_in_a_big_enough_cluster_is_scored(anomalies):
    features, result = anomalies
    assert len(result.scores) == len(features)
    scored = result.scores["combined_score"].dropna()
    # with k=4 over 20k shops every cluster exceeds min size, so all score
    assert len(scored) == len(features)
    assert scored.between(0, 1).all()


def test_top_anomalies_is_top_decile(anomalies):
    features, result = anomalies
    n_scored = result.scores["combined_score"].notna().sum()
    assert len(result.top_anomalies) == max(1, int(round(n_scored * 0.10)))
    # top of the ranking really is the max combined score
    assert (
        result.top_anomalies["combined_score"].iloc[0]
        == result.scores["combined_score"].max()
    )
    # descending order
    diffs = np.diff(result.top_anomalies["combined_score"].to_numpy())
    assert (diffs <= 1e-12).all()


def test_top_anomalies_carry_triage_context(anomalies):
    _, result = anomalies
    for col in ("division", "shop_type", "cluster", "latitude", "longitude"):
        assert col in result.top_anomalies.columns


def test_small_clusters_skipped_gracefully(shops_sample, targets_sample, config):
    features, _ = run_preprocessing(
        shops_sample.head(2_000), targets_sample, config
    )
    cfg = Config(
        kmeans_k_range=(4, 4),
        hierarchical_sample_size=1_000,
        anomaly_min_cluster_size=2_000,  # force every cluster below the floor
    )
    clusters = run_clustering(features, cfg)
    result = run_anomaly(features, clusters.labels, cfg)
    assert result.n_skipped_small_clusters >= 1
    assert result.scores["combined_score"].isna().any()
