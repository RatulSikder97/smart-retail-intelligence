import json

import pytest

from smart_retail.config import Config
from smart_retail import pipeline


@pytest.fixture(scope="module")
def pipeline_run(config, tmp_path_factory):
    """Full 5-phase run on a 3k-shop slice of the real data, artifacts in tmp."""
    tmp = tmp_path_factory.mktemp("pipeline")
    cfg = Config(
        raw_dir=config.raw_dir,
        data_dir=tmp / "data",
        models_dir=tmp / "models",
        reports_dir=tmp / "reports",
        kmeans_k_range=(3, 5),
        hierarchical_sample_size=2_000,
        shap_sample_size=300,
    )
    # 10k rows: the smallest slice where a healthy share of shops (~60%)
    # has valued targets — the first ~6k shop rows map to all-NaN targets
    metrics = pipeline.run([1, 2, 3, 4, 5], cfg, nrows=10_000)
    return cfg, metrics


def test_all_artifacts_written(pipeline_run):
    cfg, _ = pipeline_run
    for name in (
        pipeline.FEATURES_PARQUET,
        pipeline.CLUSTERS_PARQUET,
        pipeline.PREDICTIONS_PARQUET,
        pipeline.RISK_PARQUET,
        pipeline.ANOMALIES_PARQUET,
        pipeline.TOP_ANOMALIES_PARQUET,
    ):
        assert (cfg.processed_dir / name).exists(), name
    assert (cfg.models_dir / "clustering_kmeans.joblib").exists()
    assert (cfg.models_dir / "regression_best.joblib").exists()
    assert (cfg.models_dir / "classification_best.joblib").exists()
    assert (cfg.reports_dir / "cluster_profiles.csv").exists()
    assert (cfg.reports_dir / "shap_importance.csv").exists()


def test_metrics_json_covers_all_phases(pipeline_run):
    cfg, metrics = pipeline_run
    on_disk = json.loads((cfg.reports_dir / pipeline.METRICS_JSON).read_text())
    for key in (
        "phase_1_preprocessing",
        "phase_2_clustering",
        "phase_3_regression",
        "phase_4_classification",
        "phase_5_anomaly",
        "durations_sec",
    ):
        assert key in on_disk, key
    assert on_disk["phase_2_clustering"]["best_k"] in (3, 4, 5)
    assert set(on_disk["durations_sec"]) == {f"phase_{i}" for i in range(1, 6)}
    assert metrics.keys() == on_disk.keys()


def test_phases_can_resume_from_disk(pipeline_run):
    """Phase 5 in a fresh invocation reloads features + clusters from parquet."""
    cfg, _ = pipeline_run
    metrics = pipeline.run([5], cfg)
    assert metrics["phase_5_anomaly"]["n_scored"] > 0


def test_unknown_phase_rejected(pipeline_run):
    cfg, _ = pipeline_run
    with pytest.raises(ValueError, match="Unknown phase"):
        pipeline.run([9], cfg)
