"""Dataset profiling: per-column statistics for the three raw files, written once to reports/dataset_profile.json."""

from __future__ import annotations

import numpy as np
import pandas as pd

from smart_retail.preprocessing import schema

# gap-handling notes shown verbatim on the dashboard's dataset pages
_CATEGORICAL_FILL = (
    "Missing in the source (never captured at survey time). Phase 1 keeps the "
    "row and fills the gap with an explicit 'Unknown' level so these shops "
    "stay analysable — dropping them would discard shops that DO have GPS "
    "and targets. 'Unknown' then behaves as its own category in clustering "
    "and the models."
)
TRANSFORMATIONS = {
    "organizer": _CATEGORICAL_FILL,
    "place_nature": _CATEGORICAL_FILL,
    "place_name": "Free text; not used as a model feature.",
    "freq_type": _CATEGORICAL_FILL,
    "shop_type": _CATEGORICAL_FILL,
    "shop_style": _CATEGORICAL_FILL,
    "shop_importance": _CATEGORICAL_FILL,
    "ss_importance": _CATEGORICAL_FILL,
    "visit_target": "99.7% missing in the source — dropped, carries no usable signal.",
    "latitude": "Rows with missing/out-of-country GPS are removed in phase 1 "
                "(13.6%; geometry = 'POINT EMPTY'). No imputation — a made-up "
                "location would poison every spatial feature.",
    "longitude": "Same rule as latitude.",
    "geometry": "Redundant WKT copy of latitude/longitude — dropped after GPS validation.",
    "zila": _CATEGORICAL_FILL,
    "upazila": _CATEGORICAL_FILL,
    "division": _CATEGORICAL_FILL,
    "Total": "NaN means the shop has no assigned target (rows ~736k+ of the "
             "file). Kept as NaN and excluded from the supervised phases "
             "(regression/classification) — never imputed, because an invented "
             "target would become a fake label.",
    **{f"prod_{i}": "NaN treated as 0 when counting a shop's targeted products "
                    "(n_products_targeted)." for i in range(1, 18)},
    "price_1": "Importance-tier price; missing for SKUs not sold in that tier — left as-is.",
    "price_2": "Importance-tier price; missing for SKUs not sold in that tier — left as-is.",
    "price_3": "Importance-tier price; missing for SKUs not sold in that tier — left as-is.",
    "ingredient": "Missing for ~29% of SKUs in the source catalogue — left as-is.",
}


def profile_columns(df: pd.DataFrame, descriptions: dict[str, str] | None = None) -> list[dict]:
    descriptions = descriptions or {}
    out = []
    for col in df.columns:
        s = df[col]
        examples = s.dropna().unique()[:3]
        record = {
            "column": col,
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "missing": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean()) * 100, 2),
            "unique": int(s.nunique()),
            "examples": ", ".join(str(v)[:24] for v in examples),
            "description": descriptions.get(col, ""),
            "transformation": TRANSFORMATIONS.get(col, ""),
        }
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            clean = s.dropna()
            mode = clean.mode()
            record.update(
                kind="numeric",
                min=round(float(clean.min()), 3),
                q25=round(float(clean.quantile(0.25)), 3),
                median=round(float(clean.median()), 3),
                mean=round(float(clean.mean()), 3),
                mode=round(float(mode.iloc[0]), 3) if not mode.empty else None,
                q75=round(float(clean.quantile(0.75)), 3),
                p95=round(float(clean.quantile(0.95)), 3),
                max=round(float(clean.max()), 3),
                std=round(float(clean.std()), 3),
                skew=round(float(clean.skew()), 3) if len(clean) > 2 else None,
            )
            # histogram is clipped at p99 so one giant outlier doesn't flatten the view
            cap = float(clean.quantile(0.99))
            clipped = clean[clean <= cap]
            if clipped.nunique() > 1:
                counts, edges = np.histogram(clipped, bins=30)
                record["histogram"] = {
                    "counts": counts.tolist(),
                    "edges": [round(float(e), 4) for e in edges],
                    "clipped_at_p99": cap,
                }
        else:
            vc = s.fillna("<missing>").astype(str).value_counts()
            record.update(
                kind="categorical",
                top_values=[
                    {"value": str(v)[:40], "count": int(c)}
                    for v, c in vc.head(15).items()
                ],
                most_common=str(vc.index[0])[:40] if len(vc) else None,
                most_common_count=int(vc.iloc[0]) if len(vc) else 0,
                least_common=str(vc.index[-1])[:40] if len(vc) else None,
                least_common_count=int(vc.iloc[-1]) if len(vc) else 0,
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
