import pytest

from smart_retail.config import Config
from smart_retail.data import loader, schema


def test_shops_have_expected_schema(shops_sample):
    assert list(shops_sample.columns) == schema.SHOP_COLUMNS
    assert shops_sample[schema.SHOP_ID].notna().all()


def test_products_have_expected_schema(products):
    assert set(schema.PRODUCT_COLUMNS) <= set(products.columns)
    assert len(products) == 42
    assert products["sku_id"].is_unique


def test_targets_have_expected_schema(targets_sample):
    assert set(schema.TARGET_COLUMNS) == set(targets_sample.columns)
    assert (targets_sample[schema.TOTAL_TARGET].dropna() >= 0).all()


def test_loader_raises_with_download_hint_when_file_missing(tmp_path):
    cfg = Config(raw_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="mendeley"):
        loader.load_shops(cfg)


def test_loader_rejects_wrong_columns(tmp_path):
    bad = tmp_path / schema.SHOP_FILE
    bad.write_text("a,b\n1,2\n")
    cfg = Config(raw_dir=tmp_path)
    with pytest.raises(ValueError, match="missing expected columns"):
        loader.load_shops(cfg)
