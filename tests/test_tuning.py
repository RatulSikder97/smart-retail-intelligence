import pytest

from smart_retail.config import Config
from smart_retail.preprocessing.features import run_preprocessing
from smart_retail.modeling.classification import run_classification
from smart_retail.modeling.clustering import run_clustering
from smart_retail.modeling.regression import run_regression


@pytest.fixture(scope="module")
def tuned_setup(shops_sample, targets_sample, config):
    features, _ = run_preprocessing(shops_sample, targets_sample, config)
    cfg = Config(
        tune=True, cv=True,
        tuning_trials=3, tuning_sample_size=5_000, cv_folds=3,
        kmeans_k_range=(4, 4), hierarchical_sample_size=2_000,
        shap_sample_size=300,
    )
    return features, cfg


def test_regression_tuning_and_cv(tuned_setup):
    features, cfg = tuned_setup
    result = run_regression(features, cfg)

    assert result.tuning is not None
    assert result.tuning["engine"].startswith("Optuna TPE")
    assert result.tuning["n_trials"] == 3
    assert set(result.tuning["best_params"]) >= {"n_estimators", "max_depth", "learning_rate"}
    assert result.tuning["best_cv_rmse"] > 0
    # tuned model competes as its own candidate; the default stays visible
    assert "xgboost_tuned" in result.metrics and "xgboost" in result.metrics

    assert result.cv is not None
    assert result.cv["folds"] == 3
    for metric in ("r2", "rmse"):
        assert set(result.cv[metric]) == {"mean", "std"}
        assert result.cv[metric]["std"] >= 0
    assert result.cv["rmse"]["mean"] > 0


def test_classification_tuning_and_cv(tuned_setup):
    features, cfg = tuned_setup
    clusters = run_clustering(features, cfg)
    result = run_classification(features, clusters.labels, cfg)

    assert result.tuning is not None
    assert 0.5 <= result.tuning["best_cv_roc_auc"] <= 1.0
    assert "class weights inside each fold" in result.tuning["objective"]

    assert result.cv is not None
    assert 0.5 <= result.cv["roc_auc"]["mean"] <= 1.0
    assert 0.0 <= result.cv["f1"]["mean"] <= 1.0


def test_tuning_off_by_default(shops_sample, targets_sample, config):
    features, _ = run_preprocessing(shops_sample, targets_sample, config)
    cfg = Config(shap_sample_size=300)
    result = run_regression(features, cfg)
    assert result.tuning is None and result.cv is None
