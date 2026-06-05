"""Phase 3 (RQ2) — predict a shop's monthly sales target from its location, type, importance, density, and assortment breadth."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from smart_retail.config import Config
from smart_retail.preprocessing import schema
from smart_retail.preprocessing.features import TARGET_COLUMN, feature_matrix


@dataclass
class RegressionResult:
    metrics: dict                 # model name -> {rmse, mae, r2}
    best_model_name: str
    best_model: object
    shap_importance: pd.DataFrame  # feature, mean_abs_shap — from best tree model
    predictions: pd.DataFrame      # shop_code, actual, predicted
    feature_names: list[str] = field(default_factory=list)
    tuning: dict | None = None     # Optuna study summary (config.tune)
    cv: dict | None = None         # k-fold mean ± std of the best model (config.cv)
    cv_tuned: dict | None = None   # same CV for the tuned candidate, if distinct


def _candidate_models(config: Config) -> dict[str, object]:
    return {
        "linear_regression": make_pipeline(StandardScaler(), LinearRegression()),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=5,
            n_jobs=config.n_jobs,
            random_state=config.random_state,
        ),
        "xgboost": XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            n_jobs=config.n_jobs,
            random_state=config.random_state,
            verbosity=0,
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=300,
            num_leaves=63,
            learning_rate=0.1,
            n_jobs=config.n_jobs,
            random_state=config.random_state,
            verbose=-1,
        ),
    }


def run_regression(features: pd.DataFrame, config: Config) -> RegressionResult:
    labelled = features[features["has_target"]]
    if labelled.empty:
        raise ValueError(
            "No shops with a sales target in this slice — targets only cover "
            "~86% of shops and small --nrows slices can miss them entirely. "
            "Increase --nrows or run on the full dataset."
        )
    X = feature_matrix(labelled)
    y = labelled[TARGET_COLUMN].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )

    candidates = _candidate_models(config)
    tuning_summary = None
    if config.tune:
        from smart_retail.modeling.tuning import tune_xgb_regressor

        tuning_summary = tune_xgb_regressor(X_train, y_train, config)
        # separate candidate: the default-vs-tuned comparison stays visible
        candidates["xgboost_tuned"] = XGBRegressor(**{
            **tuning_summary["best_params"],
            "tree_method": "hist",
            "n_jobs": config.n_jobs,
            "random_state": config.random_state,
            "verbosity": 0,
        })

    metrics: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics[name] = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "mae": float(mean_absolute_error(y_test, pred)),
            "r2": float(r2_score(y_test, pred)),
        }
        fitted[name] = model

    best_model_name = min(metrics, key=lambda name: metrics[name]["rmse"])
    best_model = fitted[best_model_name]

    cv_summary, cv_tuned_summary = None, None
    if config.cv:
        from sklearn.base import clone

        from smart_retail.modeling.tuning import cv_regression

        cv_summary = cv_regression(lambda: clone(best_model), X, y, config)
        # tuned candidate gets its own CV — the split winner can be fold-unstable on the heavy-tailed target
        if "xgboost_tuned" in fitted and best_model_name != "xgboost_tuned":
            cv_tuned_summary = cv_regression(
                lambda: clone(fitted["xgboost_tuned"]), X, y, config
            )

    # TreeExplainer can't handle the linear baseline — fall back to the best-RMSE tree model
    tree_names = [n for n in fitted if n != "linear_regression"]
    shap_model_name = (
        best_model_name if best_model_name in tree_names
        else min(tree_names, key=lambda name: metrics[name]["rmse"])
    )
    shap_sample = X_test.sample(
        min(config.shap_sample_size, len(X_test)), random_state=config.random_state
    )
    explainer = shap.TreeExplainer(fitted[shap_model_name])
    shap_values = explainer.shap_values(shap_sample)
    shap_importance = (
        pd.DataFrame(
            {
                "feature": shap_sample.columns,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    predictions = pd.DataFrame(
        {
            schema.SHOP_ID: labelled[schema.SHOP_ID].values,
            "actual": y.values,
            "predicted": best_model.predict(X),
        }
    )

    return RegressionResult(
        metrics=metrics,
        best_model_name=best_model_name,
        best_model=best_model,
        shap_importance=shap_importance,
        predictions=predictions,
        feature_names=list(X.columns),
        tuning=tuning_summary,
        cv=cv_summary,
        cv_tuned=cv_tuned_summary,
    )
