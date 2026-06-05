"""Shared fixtures. All tests run against slices of the REAL Mendeley dataset.

If the dataset is not present (see README), the whole suite is skipped with a
download hint rather than failing.
"""

from __future__ import annotations

import pytest

from smart_retail.config import Config
from smart_retail.preprocessing import schema, loader


@pytest.fixture(scope="session")
def config() -> Config:
    cfg = Config()
    for filename in (schema.SHOP_FILE, schema.PRODUCT_FILE, schema.TARGET_FILE):
        if not (cfg.raw_dir / filename).exists():
            pytest.skip(
                f"Real dataset file {filename} missing from {cfg.raw_dir} — "
                "download it from https://data.mendeley.com/datasets/mgzvngzng2/1",
                allow_module_level=True,
            )
    return cfg


@pytest.fixture(scope="session")
def shops_sample(config: Config):
    """First 20k real shop records — enough to exercise dedup/GPS/density logic."""
    return loader.load_shops(config, nrows=20_000)


@pytest.fixture(scope="session")
def targets_sample(config: Config, shops_sample):
    """Real target rows for the sampled shops.

    The two CSVs are ordered differently, so slicing both by row number
    yields zero overlap — load the full (89 MB) targets file once per
    session and filter it to the sampled shop codes instead.
    """
    targets = loader.load_targets(config)
    return targets[targets[schema.SHOP_ID].isin(set(shops_sample[schema.SHOP_ID]))]


@pytest.fixture(scope="session")
def products(config: Config):
    return loader.load_products(config)
