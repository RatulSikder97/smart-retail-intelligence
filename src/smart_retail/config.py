"""Central configuration for paths and model tunables.

Every path can be overridden with environment variables so the same code
runs on the host and inside Docker (where volumes are mounted at /app/*).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # --- paths ---
    raw_dir: Path = PROJECT_ROOT / "dataset"
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"

    # --- phase 1: preprocessing / features ---
    density_radii_km: tuple[float, ...] = (0.5, 1.0, 5.0)
    # k-NN imputation (sensitivity experiment; off by default so the
    # reference run keeps missingness explicit — see README):
    # geo fields are always safe (~99.8% accurate); shop_style only when
    # >= impute_knn_confidence of neighbours agree; shop_type/importance
    # are never imputed (measured at/below the majority-class baseline)
    impute_knn: bool = False
    impute_knn_k: int = 10
    impute_knn_confidence: float = 0.7

    # --- phase 2: clustering ---
    kmeans_k_range: tuple[int, int] = (3, 9)  # inclusive
    hierarchical_sample_size: int = 20_000
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 50

    # --- phase 3: regression ---
    test_size: float = 0.2
    shap_sample_size: int = 2_000

    # --- Bayesian tuning + cross-validation (phases 3 & 4) ---
    tune: bool = False            # Optuna TPE search for the XGBoost candidate
    tuning_trials: int = 30
    tuning_sample_size: int = 100_000
    cv: bool = False              # 5-fold CV of the best model (mean ± std)
    cv_folds: int = 5

    # --- phase 4: classification ---
    risk_threshold: float = 0.8  # at-risk if Total < 0.8 * peer-group median

    # --- phase 5: anomaly detection ---
    anomaly_top_decile: float = 0.10
    anomaly_min_cluster_size: int = 20

    # --- global ---
    random_state: int = 42
    n_jobs: int = -1

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
        if raw_dir := os.environ.get("SMART_RETAIL_RAW_DIR"):
            cfg.raw_dir = Path(raw_dir)
        if data_dir := os.environ.get("SMART_RETAIL_DATA_DIR"):
            cfg.data_dir = Path(data_dir)
        if models_dir := os.environ.get("SMART_RETAIL_MODELS_DIR"):
            cfg.models_dir = Path(models_dir)
        if reports_dir := os.environ.get("SMART_RETAIL_REPORTS_DIR"):
            cfg.reports_dir = Path(reports_dir)
        return cfg

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.processed_dir, self.models_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)
