import json

import pandas as pd
import pytest

from smart_retail import reporting
from smart_retail.config import Config


@pytest.fixture(scope="module")
def real_metrics(config: Config):
    """Metrics from the latest real pipeline run, if present."""
    path = Config().reports_dir / "metrics.json"
    if not path.exists():
        pytest.skip("no metrics.json — run the pipeline first")
    return json.loads(path.read_text())


def test_build_report_covers_all_sections(real_metrics):
    sections = reporting.build_report(real_metrics)
    assert list(sections) == [
        "data_quality", "feature_engineering",
        "rq1", "rq2", "rq3", "rq4", "conclusion",
    ]
    for key, md in sections.items():
        assert md.strip(), key


def test_feature_engineering_section_lists_feature_groups(real_metrics):
    md = reporting.feature_engineering_section(real_metrics)
    n_features = real_metrics["phase_1_preprocessing"]["n_feature_columns"]
    assert f"{n_features} model features" in md
    for feature in ("density_500m", "dist_to_district_centre_km", "n_products_targeted"):
        assert feature in md


def test_conclusion_summarises_end_to_end(real_metrics):
    md = reporting.conclusion_section(real_metrics)
    assert f"{real_metrics['phase_1_preprocessing']['n_raw_shop_rows']:,}" in md
    assert f"{real_metrics['phase_5_anomaly']['n_top_anomalies']:,}" in md
    assert "Recommended actions" in md


def test_data_quality_section_explains_missing_gps(real_metrics):
    md = reporting.data_quality_section(real_metrics)
    q = real_metrics["phase_1_preprocessing"]
    assert f"{q['n_raw_shop_rows']:,}" in md
    assert f"{q['n_invalid_or_missing_gps']:,}" in md
    assert "POINT EMPTY" in md


def test_rq_sections_carry_key_numbers(real_metrics):
    rq1 = reporting.rq1_section(real_metrics, profiles=None)
    assert f"Best k = {real_metrics['phase_2_clustering']['best_k']}" in rq1

    rq2 = reporting.rq2_section(real_metrics, shap_importance=None)
    assert real_metrics["phase_3_regression"]["best_model"] in rq2

    risk_by_division = pd.Series(
        {"Khulna": 0.485, "Barisal": 0.278, "Unknown": 0.08}
    )
    rq3 = reporting.rq3_section(real_metrics, risk_by_division)
    assert "48.5% in Khulna" in rq3
    assert "27.8% in Barisal" in rq3
    assert "Unknown" not in rq3.split("prioritisation")[0].split("uneven")[-1]

    rq4 = reporting.rq4_section(real_metrics, top_anomalies=None)
    n_top = real_metrics["phase_5_anomaly"]["n_top_anomalies"]
    assert f"{n_top:,}" in rq4


def test_report_markdown_is_one_document(real_metrics):
    sections = reporting.build_report(real_metrics)
    doc = reporting.report_markdown(sections)
    assert doc.startswith("# Smart Retail Intelligence")
    for md in sections.values():
        assert md in doc
