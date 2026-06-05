from smart_retail.preprocessing import schema
from smart_retail.exploration.profiling import build_dataset_profile, profile_columns


def test_profile_columns_covers_every_column(shops_sample):
    records = profile_columns(shops_sample)
    assert [r["column"] for r in records] == list(shops_sample.columns)
    lat = next(r for r in records if r["column"] == "latitude")
    assert 0 <= lat["missing_pct"] <= 100
    assert "median" in lat  # numeric columns carry summary stats
    code = next(r for r in records if r["column"] == "shop_code")
    assert code["unique"] > 0 and code["examples"]


def test_build_dataset_profile_structure(shops_sample, targets_sample, products):
    profile = build_dataset_profile(shops_sample, targets_sample, products)
    assert set(profile["files"]) == {schema.SHOP_FILE, schema.TARGET_FILE, schema.PRODUCT_FILE}
    assert profile["files"][schema.SHOP_FILE]["rows"] == len(shops_sample)
    assert profile["catalogue_summary"]["skus"] == 42
    ts = profile["target_summary"]
    assert ts["rows_with_total"] + ts["rows_all_nan"] == len(targets_sample)
    assert ts["total_min"] <= ts["total_median"] <= ts["total_p95"] <= ts["total_max"]
