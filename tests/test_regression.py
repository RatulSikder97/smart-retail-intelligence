import numpy as np
import pytest

from smart_retail.config import Config
from smart_retail.features.engineering import run_preprocessing
from smart_retail.models.regression import run_regression

MODELS = {"linear_regression", "random_forest", "xgboost", "lightgbm"}


@pytest.fixture(scope="module")
def regression(shops_sample, targets_sample, config):
    features, _ = run_preprocessing(shops_sample, targets_sample, config)
    cfg = Config(shap_sample_size=500)
    return features, run_regression(features, cfg)


def test_all_four_models_benchmarked(regression):
    _, result = regression
    assert set(result.metrics) == MODELS
    for scores in result.metrics.values():
        assert np.isfinite(scores["rmse"]) and scores["rmse"] > 0
        assert np.isfinite(scores["mae"])
        assert np.isfinite(scores["r2"])


def test_best_model_has_lowest_rmse(regression):
    _, result = regression
    best_rmse = result.metrics[result.best_model_name]["rmse"]
    assert best_rmse == min(s["rmse"] for s in result.metrics.values())


def test_shap_importance_covers_features(regression):
    _, result = regression
    assert not result.shap_importance.empty
    assert set(result.shap_importance["feature"]) == set(result.feature_names)
    assert (result.shap_importance["mean_abs_shap"] >= 0).all()
    # sorted descending
    values = result.shap_importance["mean_abs_shap"].to_numpy()
    assert (np.diff(values) <= 1e-12).all()


def test_predictions_cover_all_labelled_shops(regression):
    features, result = regression
    assert len(result.predictions) == int(features["has_target"].sum())
    assert np.isfinite(result.predictions["predicted"]).all()
