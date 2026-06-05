"""CLI orchestrator: runs phases 1-5 and persists every artifact.

Usage:
    python -m smart_retail.pipeline                 # all phases, full dataset
    python -m smart_retail.pipeline --phases 1,2    # subset of phases
    python -m smart_retail.pipeline --nrows 50000   # quick run on a row slice

Phases read their inputs from data/processed/ when run in a separate
invocation from the phase that produced them.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import joblib
import pandas as pd

from smart_retail.config import Config
from smart_retail.data import loader, schema
from smart_retail.features.engineering import run_preprocessing
from smart_retail.models.anomaly import run_anomaly
from smart_retail.models.classification import run_classification
from smart_retail.models.clustering import run_clustering
from smart_retail.models.regression import run_regression
from smart_retail.profiling import build_dataset_profile

log = logging.getLogger("smart_retail")

FEATURES_PARQUET = "features.parquet"
CLUSTERS_PARQUET = "clusters.parquet"
PREDICTIONS_PARQUET = "predictions.parquet"
RISK_PARQUET = "risk.parquet"
ANOMALIES_PARQUET = "anomalies.parquet"
TOP_ANOMALIES_PARQUET = "top_anomalies.parquet"
METRICS_JSON = "metrics.json"


def _load_features(config: Config) -> pd.DataFrame:
    path = config.processed_dir / FEATURES_PARQUET
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run phase 1 first.")
    return pd.read_parquet(path)


def _load_cluster_labels(config: Config, features: pd.DataFrame) -> pd.Series:
    path = config.processed_dir / CLUSTERS_PARQUET
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run phase 2 first.")
    clusters = pd.read_parquet(path)
    merged = features[[schema.SHOP_ID]].merge(clusters, on=schema.SHOP_ID, how="left")
    if merged["cluster"].isna().any():
        raise ValueError(
            "clusters.parquet does not cover all shops in features.parquet — "
            "re-run phases 1 and 2 together."
        )
    return pd.Series(merged["cluster"].to_numpy(), index=features.index, name="cluster")


def phase_1_preprocess(config: Config, nrows: int | None) -> tuple[pd.DataFrame, dict]:
    shops = loader.load_shops(config, nrows=nrows)
    targets = loader.load_targets(config)

    profile = build_dataset_profile(shops, targets, loader.load_products(config))
    (config.reports_dir / "dataset_profile.json").write_text(json.dumps(profile, indent=2))

    features, report = run_preprocessing(shops, targets, config)
    features.to_parquet(config.processed_dir / FEATURES_PARQUET, index=False)
    return features, report


def phase_2_cluster(config: Config, features: pd.DataFrame) -> tuple[pd.Series, dict]:
    result = run_clustering(features, config)
    pd.DataFrame(
        {schema.SHOP_ID: features[schema.SHOP_ID].values, "cluster": result.labels.values}
    ).to_parquet(config.processed_dir / CLUSTERS_PARQUET, index=False)
    result.profiles.to_csv(config.reports_dir / "cluster_profiles.csv", index=False)
    joblib.dump(
        {"model": result.model, "scaler": result.scaler},
        config.models_dir / "clustering_kmeans.joblib",
    )
    return result.labels, result.metrics


def phase_3_regress(config: Config, features: pd.DataFrame) -> dict:
    result = run_regression(features, config)
    result.predictions.to_parquet(
        config.processed_dir / PREDICTIONS_PARQUET, index=False
    )
    result.shap_importance.to_csv(
        config.reports_dir / "shap_importance.csv", index=False
    )
    joblib.dump(result.best_model, config.models_dir / "regression_best.joblib")
    out = {
        "models": result.metrics,
        "best_model": result.best_model_name,
        "n_features": len(result.feature_names),
    }
    if result.tuning:
        out["tuning"] = result.tuning
    if result.cv:
        out["cv"] = result.cv
    if result.cv_tuned:
        out["cv_tuned"] = result.cv_tuned
    return out


def phase_4_classify(
    config: Config, features: pd.DataFrame, cluster_labels: pd.Series
) -> dict:
    result = run_classification(features, cluster_labels, config)
    result.risk_scores.to_parquet(config.processed_dir / RISK_PARQUET, index=False)
    joblib.dump(result.best_model, config.models_dir / "classification_best.joblib")
    out = {
        "models": result.metrics,
        "best_model": result.best_model_name,
        "label_definition": result.label_definition,
        "label_rate": result.label_rate,
        "risk_threshold": config.risk_threshold,
    }
    if result.tuning:
        out["tuning"] = result.tuning
    if result.cv:
        out["cv"] = result.cv
    if result.cv_tuned:
        out["cv_tuned"] = result.cv_tuned
    return out


def phase_5_anomaly(
    config: Config, features: pd.DataFrame, cluster_labels: pd.Series
) -> dict:
    result = run_anomaly(features, cluster_labels, config)
    result.scores.to_parquet(config.processed_dir / ANOMALIES_PARQUET, index=False)
    result.top_anomalies.to_parquet(
        config.processed_dir / TOP_ANOMALIES_PARQUET, index=False
    )
    return {
        "n_scored": int(result.scores["combined_score"].notna().sum()),
        "n_top_anomalies": len(result.top_anomalies),
        "n_skipped_small_clusters": result.n_skipped_small_clusters,
    }


def run(phases: list[int], config: Config, nrows: int | None = None) -> dict:
    config.ensure_dirs()
    metrics_path = config.reports_dir / METRICS_JSON
    metrics: dict = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    features: pd.DataFrame | None = None
    cluster_labels: pd.Series | None = None

    for phase in phases:
        started = time.time()
        log.info("phase %d starting", phase)

        if phase == 1:
            features, report = phase_1_preprocess(config, nrows)
            metrics["phase_1_preprocessing"] = report
        elif phase == 2:
            features = _load_features(config) if features is None else features
            cluster_labels, m = phase_2_cluster(config, features)
            metrics["phase_2_clustering"] = m
        elif phase == 3:
            features = _load_features(config) if features is None else features
            metrics["phase_3_regression"] = phase_3_regress(config, features)
        elif phase == 4:
            features = _load_features(config) if features is None else features
            if cluster_labels is None:
                cluster_labels = _load_cluster_labels(config, features)
            metrics["phase_4_classification"] = phase_4_classify(
                config, features, cluster_labels
            )
        elif phase == 5:
            features = _load_features(config) if features is None else features
            if cluster_labels is None:
                cluster_labels = _load_cluster_labels(config, features)
            metrics["phase_5_anomaly"] = phase_5_anomaly(
                config, features, cluster_labels
            )
        else:
            raise ValueError(f"Unknown phase {phase}; valid phases are 1-5.")

        duration = round(time.time() - started, 1)
        metrics.setdefault("durations_sec", {})[f"phase_{phase}"] = duration
        metrics_path.write_text(json.dumps(metrics, indent=2))
        log.info("phase %d done in %.1fs", phase, duration)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Retail Intelligence pipeline")
    parser.add_argument(
        "--phases",
        default="1,2,3,4,5",
        help="Comma-separated phases to run (default: 1,2,3,4,5)",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Limit raw shop rows for a quick run (default: full dataset)",
    )
    parser.add_argument(
        "--impute-knn",
        action="store_true",
        help="Sensitivity experiment: k-NN impute geo fields + high-confidence "
             "shop_style (see README; off by default)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Bayesian (Optuna TPE) hyperparameter search for the XGBoost "
             "candidate in phases 3 and 4",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="5-fold cross-validation of the best model in phases 3 and 4 "
             "(headline metrics as mean ± std)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    phases = [int(p) for p in args.phases.split(",") if p.strip()]
    config = Config.from_env()
    if args.impute_knn:
        config.impute_knn = True
    if args.tune:
        config.tune = True
    if args.cv:
        config.cv = True
    run(phases, config, nrows=args.nrows)
    log.info("pipeline complete — metrics in %s", config.reports_dir / METRICS_JSON)


if __name__ == "__main__":
    main()
