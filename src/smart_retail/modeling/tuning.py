"""Bayesian hyperparameter tuning (Optuna TPE) and k-fold cross-validation for the XGBoost candidates."""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor

from smart_retail.config import Config

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest_xgb_params(trial: optuna.Trial, config: Config) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "tree_method": "hist",
        "n_jobs": config.n_jobs,
        "random_state": config.random_state,
        "verbosity": 0,
    }


def _subsample(X: pd.DataFrame, y: pd.Series, config: Config) -> tuple:
    if len(X) <= config.tuning_sample_size:
        return X, y
    idx = X.sample(config.tuning_sample_size, random_state=config.random_state).index
    return X.loc[idx], y.loc[idx]


def tune_xgb_regressor(X: pd.DataFrame, y: pd.Series, config: Config) -> dict:
    Xs, ys = _subsample(X, y, config)
    folds = KFold(n_splits=3, shuffle=True, random_state=config.random_state)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_xgb_params(trial, config)
        rmses = []
        for train_idx, val_idx in folds.split(Xs):
            model = XGBRegressor(**params)
            model.fit(Xs.iloc[train_idx], ys.iloc[train_idx])
            pred = model.predict(Xs.iloc[val_idx])
            rmses.append(np.sqrt(mean_squared_error(ys.iloc[val_idx], pred)))
        return float(np.mean(rmses))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.random_state),
    )
    study.optimize(objective, n_trials=config.tuning_trials, show_progress_bar=False)
    return {
        "engine": "Optuna TPE (Bayesian)",
        "n_trials": config.tuning_trials,
        "tuning_sample": len(Xs),
        "objective": "3-fold CV RMSE (minimized)",
        "best_cv_rmse": round(study.best_value, 4),
        "best_params": study.best_params,
    }


def tune_xgb_classifier(X: pd.DataFrame, y: pd.Series, config: Config) -> dict:
    Xs, ys = _subsample(X, y, config)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.random_state)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_xgb_params(trial, config)
        aucs = []
        for train_idx, val_idx in folds.split(Xs, ys):
            pipe = ImbPipeline([
                ("smote", SMOTE(random_state=config.random_state)),
                ("model", XGBClassifier(**params)),
            ])
            pipe.fit(Xs.iloc[train_idx], ys.iloc[train_idx])
            proba = pipe.predict_proba(Xs.iloc[val_idx])[:, 1]
            aucs.append(roc_auc_score(ys.iloc[val_idx], proba))
        return float(np.mean(aucs))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.random_state),
    )
    study.optimize(objective, n_trials=config.tuning_trials, show_progress_bar=False)
    return {
        "engine": "Optuna TPE (Bayesian)",
        "n_trials": config.tuning_trials,
        "tuning_sample": len(Xs),
        "objective": "3-fold CV ROC-AUC (maximized, SMOTE inside each fold)",
        "best_cv_roc_auc": round(study.best_value, 4),
        "best_params": study.best_params,
    }


def _mean_std(values: list[float]) -> dict:
    return {"mean": round(float(np.mean(values)), 4),
            "std": round(float(np.std(values)), 4)}


def cv_regression(model_factory, X: pd.DataFrame, y: pd.Series, config: Config) -> dict:
    folds = KFold(n_splits=config.cv_folds, shuffle=True,
                  random_state=config.random_state)
    r2s, rmses = [], []
    for train_idx, val_idx in folds.split(X):
        model = model_factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[val_idx])
        r2s.append(r2_score(y.iloc[val_idx], pred))
        rmses.append(np.sqrt(mean_squared_error(y.iloc[val_idx], pred)))
    return {"folds": config.cv_folds, "r2": _mean_std(r2s), "rmse": _mean_std(rmses)}


def cv_classification(pipeline_factory, X: pd.DataFrame, y: pd.Series, config: Config) -> dict:
    # factory must return a fresh (SMOTE → model) pipeline so resampling stays inside each training fold
    folds = StratifiedKFold(n_splits=config.cv_folds, shuffle=True,
                            random_state=config.random_state)
    aucs, f1s = [], []
    for train_idx, val_idx in folds.split(X, y):
        pipe = pipeline_factory()
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = pipe.predict_proba(X.iloc[val_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[val_idx], proba))
        f1s.append(f1_score(y.iloc[val_idx], (proba >= 0.5).astype(int)))
    return {"folds": config.cv_folds, "roc_auc": _mean_std(aucs), "f1": _mean_std(f1s)}
