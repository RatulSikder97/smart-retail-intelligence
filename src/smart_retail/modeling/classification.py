"""Phase 4 (RQ3) — predict which shops are at risk of missing their targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from smart_retail.config import Config
from smart_retail.preprocessing import schema
from smart_retail.preprocessing.features import TARGET_COLUMN, feature_matrix

LABEL_DEFINITION = (
    "at_risk = total_target < risk_threshold * median(total_target | cluster, shop_type)"
)


@dataclass
class ClassificationResult:
    label_definition: str
    label_rate: float
    metrics: dict                # model name -> {roc_auc, f1, precision, recall}
    best_model_name: str
    best_model: object
    risk_scores: pd.DataFrame    # shop_code, at_risk, risk_score
    tuning: dict | None = None   # Optuna study summary (config.tune)
    cv: dict | None = None       # k-fold mean ± std of the best model (config.cv)
    cv_tuned: dict | None = None # same CV for the tuned candidate, if distinct


def build_risk_labels(
    features: pd.DataFrame, cluster_labels: pd.Series, config: Config
) -> pd.Series:
    # proxy label (no realized sales in dataset): at-risk = under-indexes (cluster × shop_type) peers
    labelled = features[features["has_target"]]
    peer_median = (
        labelled.assign(cluster=cluster_labels.loc[labelled.index].values)
        .groupby(["cluster", "shop_type"])[TARGET_COLUMN]
        .transform("median")
    )
    return (labelled[TARGET_COLUMN] < config.risk_threshold * peer_median).astype(int)


def pos_weight_ratio(y_train: pd.Series) -> float:
    # negative/positive ratio drives the class weighting — real rows only, nothing resampled
    return float((y_train == 0).sum() / (y_train == 1).sum())


def _candidate_models(config: Config, pos_weight: float) -> dict:
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=config.n_jobs,
            random_state=config.random_state,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            scale_pos_weight=pos_weight,
            n_jobs=config.n_jobs,
            random_state=config.random_state,
            verbosity=0,
        ),
    }


def run_classification(
    features: pd.DataFrame, cluster_labels: pd.Series, config: Config
) -> ClassificationResult:
    labelled = features[features["has_target"]]
    if labelled.empty:
        raise ValueError(
            "No shops with a sales target in this slice — targets only cover "
            "~86% of shops and small --nrows slices can miss them entirely. "
            "Increase --nrows or run on the full dataset."
        )
    y = build_risk_labels(features, cluster_labels, config)
    X = feature_matrix(labelled)
    X["cluster"] = cluster_labels.loc[labelled.index].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size,
        random_state=config.random_state, stratify=y,
    )

    pos_weight = pos_weight_ratio(y_train)
    candidates = _candidate_models(config, pos_weight)
    tuning_summary = None
    if config.tune:
        from smart_retail.modeling.tuning import tune_xgb_classifier

        tuning_summary = tune_xgb_classifier(X_train, y_train, config)
        # separate candidate: the default-vs-tuned comparison stays visible
        candidates["xgboost_tuned"] = XGBClassifier(**{
            **tuning_summary["best_params"],
            "tree_method": "hist",
            "scale_pos_weight": pos_weight,
            "n_jobs": config.n_jobs,
            "random_state": config.random_state,
            "verbosity": 0,
        })

    metrics: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics[name] = {
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "f1": float(f1_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
        }
        fitted[name] = pipeline

    best_model_name = max(metrics, key=lambda name: metrics[name]["roc_auc"])
    best_model = fitted[best_model_name]

    cv_summary, cv_tuned_summary = None, None
    if config.cv:
        from sklearn.base import clone

        from smart_retail.modeling.tuning import cv_classification

        cv_summary = cv_classification(lambda: clone(best_model), X, y, config)
        if "xgboost_tuned" in fitted and best_model_name != "xgboost_tuned":
            cv_tuned_summary = cv_classification(
                lambda: clone(fitted["xgboost_tuned"]), X, y, config
            )

    risk_scores = pd.DataFrame(
        {
            schema.SHOP_ID: labelled[schema.SHOP_ID].values,
            "at_risk": y.values,
            "risk_score": best_model.predict_proba(X)[:, 1],
        }
    )

    return ClassificationResult(
        label_definition=LABEL_DEFINITION,
        label_rate=float(y.mean()),
        metrics=metrics,
        best_model_name=best_model_name,
        best_model=best_model,
        risk_scores=risk_scores,
        tuning=tuning_summary,
        cv=cv_summary,
        cv_tuned=cv_tuned_summary,
    )
