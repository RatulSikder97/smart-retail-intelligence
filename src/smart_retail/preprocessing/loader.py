"""Loaders for the three raw CSVs, with column validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from smart_retail.config import Config
from smart_retail.preprocessing import schema

_DOWNLOAD_HINT = (
    "Download it from https://data.mendeley.com/datasets/mgzvngzng2/1 "
    "and place the CSVs in the raw data directory (default: ./dataset)."
)


def _read_csv(path: Path, required: list[str], nrows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. {_DOWNLOAD_HINT}")
    # pin shop_code to str — an all-numeric slice would infer int64 and silently break joins
    df = pd.read_csv(path, nrows=nrows, dtype={schema.SHOP_ID: str})
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {sorted(missing)}")
    return df


def load_shops(config: Config, nrows: int | None = None) -> pd.DataFrame:
    # one row per visit record, so shop_code can repeat
    return _read_csv(config.raw_dir / schema.SHOP_FILE, schema.SHOP_COLUMNS, nrows)


def load_products(config: Config) -> pd.DataFrame:
    return _read_csv(config.raw_dir / schema.PRODUCT_FILE, schema.PRODUCT_COLUMNS)


def load_targets(config: Config, nrows: int | None = None) -> pd.DataFrame:
    return _read_csv(config.raw_dir / schema.TARGET_FILE, schema.TARGET_COLUMNS, nrows)
