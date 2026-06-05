"""Dataset profiling: per-column statistics for the three raw files.

Runs once inside pipeline phase 1 and lands in reports/dataset_profile.json,
so the dashboard can show an in-depth data dictionary without re-reading
200 MB of CSV on every page load.
"""

from __future__ import annotations

import pandas as pd

from smart_retail.data import schema


def profile_columns(df: pd.DataFrame, descriptions: dict[str, str] | None = None) -> list[dict]:
    """One record per column: dtype, completeness, cardinality, examples."""
    descriptions = descriptions or {}
    out = []
    for col in df.columns:
        s = df[col]
        examples = s.dropna().unique()[:3]
        record = {
            "column": col,
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "missing_pct": round(float(s.isna().mean()) * 100, 2),
            "unique": int(s.nunique()),
            "examples": ", ".join(str(v)[:24] for v in examples),
            "description": descriptions.get(col, ""),
        }
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            record.update(
                min=round(float(s.min()), 3),
                median=round(float(s.median()), 3),
                mean=round(float(s.mean()), 3),
                max=round(float(s.max()), 3),
            )
        out.append(record)
    return out


SHOP_DESCRIPTIONS = {
    "shop_code": "Unique shop identifier (hex-like string)",
    "organizer": "Field organizer who registered the shop (62 values)",
    "place_nature": "Nature of the location (market, roadside, …)",
    "place_name": "Free-text location name",
    "freq_type": "Visit-frequency classification",
    "shop_type": "Business type (grocery, pharmacy, … — 12 categories)",
    "shop_style": "Appearance-based class (3 styles)",
    "shop_importance": "Company importance tier (3 levels)",
    "ss_importance": "Detailed importance rating (7 levels)",
    "visit_target": "Planned monthly visits (99.7% missing in source)",
    "latitude": "GPS latitude (13.6% missing in source)",
    "longitude": "GPS longitude (13.6% missing in source)",
    "geometry": "WKT point — 'POINT EMPTY' when GPS was not captured",
    "zila": "District (64 districts)",
    "upazila": "Sub-district (515 upazilas)",
    "division": "Administrative division (8 divisions)",
}

TARGET_DESCRIPTIONS = {
    "shop_code": "Shop reference key",
    **{f"prod_{i}": f"Monthly sales target for product {i}" for i in range(1, 18)},
    "Total": "Monthly total target across all products",
}

PRODUCT_DESCRIPTIONS = {
    "sku_id": "Unique SKU variant id",
    "sku_name": "Anonymized variant name",
    "product_id": "Product category id",
    "product_name": "Anonymized base product",
    "brand": "Brand (5, anonymized)",
    "type": "Flavour/variant category (9)",
    "packet_size": "Packaging size",
    "price_a": "Price for shop style A",
    "price_b": "Price for shop style B",
    "price_c": "Price for shop style C",
    "price_1": "Price tier 1 (importance-based)",
    "price_2": "Price tier 2 (importance-based)",
    "price_3": "Price tier 3 (importance-based)",
    "ingredient": "Material composition detail",
    "freq": "Monthly sales frequency target",
    "in_packet": "Items per packet",
    "total_volume": "Product volume",
    "total_price": "Total price / sell income",
}


def build_dataset_profile(
    shops: pd.DataFrame, targets: pd.DataFrame, products: pd.DataFrame
) -> dict:
    total = targets[schema.TOTAL_TARGET]
    return {
        "files": {
            schema.SHOP_FILE: {
                "rows": len(shops),
                "columns": profile_columns(shops, SHOP_DESCRIPTIONS),
            },
            schema.TARGET_FILE: {
                "rows": len(targets),
                "columns": profile_columns(targets, TARGET_DESCRIPTIONS),
            },
            schema.PRODUCT_FILE: {
                "rows": len(products),
                "columns": profile_columns(products, PRODUCT_DESCRIPTIONS),
            },
        },
        "target_summary": {
            "rows_with_total": int(total.notna().sum()),
            "rows_all_nan": int(total.isna().sum()),
            "duplicate_shop_codes": int(targets[schema.SHOP_ID].duplicated().sum()),
            "total_min": float(total.min()),
            "total_median": float(total.median()),
            "total_mean": round(float(total.mean()), 3),
            "total_p95": float(total.quantile(0.95)),
            "total_max": float(total.max()),
        },
        "catalogue_summary": {
            "skus": len(products),
            "brands": int(products["brand"].nunique()),
            "product_types": int(products["type"].nunique()),
            "base_products": int(products["product_id"].nunique()),
        },
    }
