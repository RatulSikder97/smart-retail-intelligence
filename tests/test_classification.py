import numpy as np
import pytest

from smart_retail.config import Config
from smart_retail.preprocessing.features import run_preprocessing
from smart_retail.modeling.classification import (
    build_risk_labels,
    run_classification,
)
from smart_retail.modeling.clustering import run_clustering

MODELS = {"logistic_regression", "random_forest", "xgboost"}


@pytest.fixture(scope="module")
def setup(shops_sample, targets_sample, config):
    features, _ = run_preprocessing(shops_sample, targets_sample, config)
    cfg = Config(kmeans_k_range=(4, 4), hierarchical_sample_size=2_000)
    clusters = run_clustering(features, cfg)
    return features, clusters.labels, cfg


@pytest.fixture(scope="module")
def classification(setup):
    features, cluster_labels, cfg = setup
    return run_classification(features, cluster_labels, cfg)


def test_label_rate_is_meaningful(setup, classification):
    features, cluster_labels, cfg = setup
    labels = build_risk_labels(features, cluster_labels, cfg)
    assert 0.0 < labels.mean() < 1.0
    assert classification.label_rate == pytest.approx(labels.mean())
    assert "risk_threshold" in classification.label_definition


def test_all_models_scored(classification):
    assert set(classification.metrics) == MODELS
    for scores in classification.metrics.values():
        assert 0.0 <= scores["roc_auc"] <= 1.0
        for key in ("f1", "precision", "recall"):
            assert 0.0 <= scores[key] <= 1.0


def test_best_model_beats_coin_flip(classification):
    best = classification.metrics[classification.best_model_name]
    assert best["roc_auc"] > 0.5


def test_risk_scores_are_probabilities_for_all_labelled_shops(setup, classification):
    features, _, _ = setup
    scores = classification.risk_scores
    assert len(scores) == int(features["has_target"].sum())
    assert scores["risk_score"].between(0, 1).all()
    assert set(scores["at_risk"].unique()) <= {0, 1}


def test_smote_never_sees_test_data(setup):
    """The pipeline resamples only inside fit(); scoring uses the raw test set."""
    features, cluster_labels, cfg = setup
    from smart_retail.modeling.classification import _candidate_models
    from smart_retail.preprocessing.features import feature_matrix
    from sklearn.model_selection import train_test_split

    labelled = features[features["has_target"]]
    y = build_risk_labels(features, cluster_labels, cfg)
    X = feature_matrix(labelled)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    X_test_before = X_test.copy()
    pipeline = _candidate_models(cfg)["logistic_regression"]
    pipeline.fit(X_train, y_train)
    pipeline.predict_proba(X_test)
    # test set untouched, and its size unchanged by resampling
    assert X_test.equals(X_test_before)
    assert len(X_test) == int(round(cfg.test_size * len(X)))
