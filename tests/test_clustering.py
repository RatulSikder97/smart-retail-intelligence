import pytest

from smart_retail.config import Config
from smart_retail.preprocessing.features import run_preprocessing
from smart_retail.modeling.clustering import run_clustering


@pytest.fixture(scope="module")
def clustering(shops_sample, targets_sample, config):
    features, _ = run_preprocessing(shops_sample, targets_sample, config)
    # narrower k range + smaller cross-check/stability samples keep the test quick
    cfg = Config(kmeans_k_range=(3, 5), hierarchical_sample_size=3_000,
                 stability_bootstraps=3, stability_sample_size=3_000)
    return features, run_clustering(features, cfg)


def test_every_shop_gets_a_cluster(clustering):
    features, result = clustering
    assert len(result.labels) == len(features)
    assert result.labels.notna().all()


def test_best_k_within_range_and_scored(clustering):
    _, result = clustering
    assert 3 <= result.best_k <= 5
    km = result.metrics["kmeans"]
    assert set(km) == {"3", "4", "5"}
    for scores in km.values():
        assert scores["silhouette"] == scores["silhouette"]  # not NaN
        assert scores["davies_bouldin"] > 0


def test_all_three_algorithms_reported(clustering):
    _, result = clustering
    assert result.metrics["chosen_algorithm"] == "kmeans"
    assert "silhouette" in result.metrics["agglomerative"]
    assert "n_clusters_found" in result.metrics["dbscan"]


def test_profiles_cover_every_cluster(clustering):
    _, result = clustering
    assert len(result.profiles) == result.best_k
    assert result.profiles["n_shops"].sum() == len(result.labels)
    assert "top_division" in result.profiles.columns


def test_bootstrap_stability_reported(clustering):
    _, result = clustering
    st = result.metrics["stability"]
    assert st["n_bootstraps"] == 3
    assert -1.0 <= st["ari_min"] <= st["ari_mean"] <= 1.0
    assert st["ari_std"] >= 0.0
    # one recovery score per archetype, each a valid Jaccard
    assert len(st["per_cluster_jaccard"]) == result.best_k
    assert all(0.0 <= v <= 1.0 for v in st["per_cluster_jaccard"].values())


def test_deterministic_under_seed(clustering, shops_sample, targets_sample, config):
    features, first = clustering
    cfg = Config(kmeans_k_range=(3, 5), hierarchical_sample_size=3_000,
                 stability_bootstraps=3, stability_sample_size=3_000)
    second = run_clustering(features, cfg)
    assert first.best_k == second.best_k
    assert (first.labels == second.labels).all()
