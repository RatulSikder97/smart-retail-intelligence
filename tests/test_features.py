import numpy as np

import pytest

from smart_retail.data import schema
from smart_retail.features.engineering import (
    TARGET_COLUMN,
    feature_matrix,
    run_preprocessing,
)


@pytest.fixture(scope="module")
def preprocessed(shops_sample, targets_sample, config):
    return run_preprocessing(shops_sample, targets_sample, config)


def test_one_row_per_shop(preprocessed):
    features, _ = preprocessed
    assert features[schema.SHOP_ID].is_unique


def test_all_gps_inside_bangladesh(preprocessed):
    features, report = preprocessed
    lat_min, lat_max, lon_min, lon_max = schema.BD_BBOX
    assert features["latitude"].between(lat_min, lat_max).all()
    assert features["longitude"].between(lon_min, lon_max).all()
    assert report["n_invalid_or_missing_gps"] >= 0
    assert report["n_final_shops"] == len(features)


def test_densities_are_monotonic_in_radius(preprocessed):
    features, _ = preprocessed
    assert (features["density_1km"] >= features["density_500m"]).all()
    assert (features["density_5km"] >= features["density_1km"]).all()
    assert (features["density_500m"] >= 0).all()


def test_district_centre_distance_is_finite_and_nonnegative(preprocessed):
    features, _ = preprocessed
    dist = features["dist_to_district_centre_km"]
    assert np.isfinite(dist).all()
    assert (dist >= 0).all()
    # every district has shops on both sides of its centroid, so mean > 0
    assert dist.mean() > 0


def test_target_join(preprocessed):
    features, report = preprocessed
    joined = features[features["has_target"]]
    assert report["n_shops_with_target"] == len(joined)
    assert joined[TARGET_COLUMN].notna().all()
    assert (joined["n_products_targeted"] >= 0).all()


def test_feature_matrix_is_numeric_and_nan_free(preprocessed):
    features, _ = preprocessed
    X = feature_matrix(features)
    assert len(X) == len(features)
    assert X.select_dtypes(exclude=[np.number]).empty
    assert not X.isna().any().any()
    assert TARGET_COLUMN not in X.columns
    assert schema.SHOP_ID not in X.columns


def test_knn_imputation_fills_geo_and_confident_style_only(shops_sample, targets_sample, config):
    from smart_retail.config import Config

    cfg = Config(impute_knn=True)
    features, report = run_preprocessing(shops_sample, targets_sample, cfg)
    imp = report["knn_imputation"]

    # business attributes stay untouched: Unknown level survives, no flag column
    assert "shop_type" in imp["never_imputed"]
    assert "shop_type_was_imputed" not in features.columns

    # geo fields: every accepted fill is flagged, flags match the report
    for col in ("zila", "upazila", "division"):
        flag = f"{col}_was_imputed"
        assert flag in features.columns
        assert int(features[flag].sum()) == imp["imputed"][col]
    # style honours the confidence threshold: anything below it stays Unknown
    assert imp["imputed"]["shop_style"] >= 0
    assert (features.loc[features["shop_style_was_imputed"] == 1, "shop_style"] != "Unknown").all()


def test_knn_imputation_off_by_default(shops_sample, targets_sample, config):
    features, report = run_preprocessing(shops_sample, targets_sample, config)
    assert "knn_imputation" not in report
    assert not any(c.endswith("_was_imputed") for c in features.columns)
