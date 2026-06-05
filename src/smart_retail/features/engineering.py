"""Phase 1 — preprocessing & feature engineering.

Input : raw shop and target frames (real Mendeley schema).
Output: one clean row per shop with GPS-validated coordinates, spatial
        density/distance features, one-hot structural features, and the
        joined monthly sales target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from smart_retail.config import Config
from smart_retail.data import schema

EARTH_RADIUS_KM = 6371.0

# Low-cardinality categoricals worth one-hot encoding as model features
ONEHOT_COLUMNS = ["shop_type", "shop_style", "shop_importance", "place_nature", "division"]

# Columns kept for identification/reporting but excluded from model matrices
ID_COLUMNS = [schema.SHOP_ID, "zila", "upazila", "division", "shop_type",
              "shop_style", "shop_importance", "place_nature"]

TARGET_COLUMN = "total_target"


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized great-circle distance in kilometres."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _shop_density(shops: pd.DataFrame, radii_km: tuple[float, ...]) -> pd.DataFrame:
    """Number of neighbouring shops within each radius (excluding the shop itself)."""
    coords = np.radians(shops[["latitude", "longitude"]].to_numpy())
    tree = BallTree(coords, metric="haversine")
    out = {}
    for radius in radii_km:
        counts = tree.query_radius(coords, r=radius / EARTH_RADIUS_KM, count_only=True)
        label = f"density_{int(radius * 1000)}m" if radius < 1 else f"density_{radius:g}km"
        out[label] = counts - 1  # query point counts itself
    return pd.DataFrame(out, index=shops.index)


def _dist_to_district_centre(shops: pd.DataFrame) -> pd.Series:
    """Distance from each shop to the centroid of its district's (zila's) shops."""
    centroids = shops.groupby("zila")[["latitude", "longitude"]].transform("mean")
    return pd.Series(
        _haversine_km(
            shops["latitude"], shops["longitude"],
            centroids["latitude"], centroids["longitude"],
        ),
        index=shops.index,
        name="dist_to_district_centre_km",
    )


def _impute_knn(shops: pd.DataFrame, config: Config, report: dict) -> pd.DataFrame:
    """Nearest-neighbour imputation for the columns where it is justified.

    Geo-administrative fields are location-determined (measured 99.8%
    accurate on held-out known shops); shop_style carries real spatial
    signal (96.5% vs 65.2% baseline) and is filled only when neighbours
    agree strongly. shop_type and shop_importance are deliberately NOT
    imputed: measured at +11pts and -1pt against the majority-class
    baseline respectively, and their joint missingness IS the ghost-shop
    signal the analysis needs to preserve.
    """
    coords = np.radians(shops[["latitude", "longitude"]].to_numpy())
    always = ["zila", "upazila", "division"]
    conditional = ["shop_style"]
    imputed_counts: dict[str, int] = {}

    for col in always + conditional:
        missing_mask = shops[col].isna().to_numpy()
        if not missing_mask.any():
            shops[f"{col}_was_imputed"] = np.zeros(len(shops), dtype=np.uint8)
            imputed_counts[col] = 0
            continue
        known_mask = ~shops[col].isna().to_numpy()
        tree = BallTree(coords[known_mask], metric="haversine")
        _, idx = tree.query(coords[missing_mask], k=config.impute_knn_k)
        neighbour_values = shops.loc[known_mask, col].to_numpy()[idx]

        votes = pd.DataFrame(neighbour_values)
        majority = votes.mode(axis=1)[0].to_numpy()
        agreement = (neighbour_values == majority[:, None]).mean(axis=1)
        threshold = config.impute_knn_confidence if col in conditional else 0.5

        fill = np.where(agreement >= threshold, majority, None)
        flag = np.zeros(len(shops), dtype=np.uint8)
        target_rows = shops.index[missing_mask]
        accepted = pd.notna(fill)
        shops.loc[target_rows[accepted], col] = fill[accepted]
        flag[np.flatnonzero(missing_mask)[accepted]] = 1
        shops[f"{col}_was_imputed"] = flag
        imputed_counts[col] = int(accepted.sum())

    report["knn_imputation"] = {
        "k": config.impute_knn_k,
        "confidence_threshold": config.impute_knn_confidence,
        "imputed": imputed_counts,
        "never_imputed": ["shop_type", "shop_importance"],
    }
    return shops


def run_preprocessing(
    shops: pd.DataFrame, targets: pd.DataFrame, config: Config
) -> tuple[pd.DataFrame, dict]:
    """Clean the raw shop records and build the per-shop feature table."""
    report: dict = {"n_raw_shop_rows": len(shops)}

    # 1. one row per shop
    shops = shops.drop_duplicates(subset=schema.SHOP_ID, keep="first").copy()
    report["n_unique_shops"] = len(shops)

    # 2. GPS validation: coordinates must exist and fall inside Bangladesh
    lat_min, lat_max, lon_min, lon_max = schema.BD_BBOX
    valid_gps = (
        shops["latitude"].between(lat_min, lat_max)
        & shops["longitude"].between(lon_min, lon_max)
    )
    report["n_invalid_or_missing_gps"] = int((~valid_gps).sum())
    shops = shops[valid_gps].copy()

    # 2b. optional k-NN imputation (geo fields + high-confidence style);
    # runs before the Unknown fill so only justified gaps get values
    if config.impute_knn:
        shops = _impute_knn(shops, config, report)

    # 3. categorical gaps → explicit "Unknown" level
    for col in schema.SHOP_CATEGORICAL + ["zila", "upazila", "division"]:
        shops[col] = shops[col].fillna("Unknown")

    # 4. spatial features
    shops = pd.concat(
        [shops, _shop_density(shops, config.density_radii_km)], axis=1
    )
    shops["dist_to_district_centre_km"] = _dist_to_district_centre(shops)

    # 5. monthly sales target join (targets file covers ~85% of shops).
    # ~2.4% of codes appear twice, sometimes once with values and once all-NaN
    # — sort valued rows first so dedup keeps the informative one.
    targets = (
        targets.sort_values(schema.TOTAL_TARGET, na_position="last")
        .drop_duplicates(subset=schema.SHOP_ID, keep="first")
    )
    target_cols = targets[[schema.SHOP_ID, schema.TOTAL_TARGET]].rename(
        columns={schema.TOTAL_TARGET: TARGET_COLUMN}
    )
    n_products = (targets[schema.PRODUCT_TARGET_COLUMNS].fillna(0) > 0).sum(axis=1)
    target_cols = target_cols.assign(n_products_targeted=n_products.values)
    shops = shops.merge(target_cols, on=schema.SHOP_ID, how="left")
    shops["has_target"] = shops[TARGET_COLUMN].notna()
    shops["n_products_targeted"] = shops["n_products_targeted"].fillna(0).astype(int)
    report["n_shops_with_target"] = int(shops["has_target"].sum())

    # 6. one-hot structural features (raw categoricals kept for reporting)
    dummies = pd.get_dummies(
        shops[ONEHOT_COLUMNS], prefix=ONEHOT_COLUMNS, dtype=np.uint8
    )
    features = pd.concat(
        [shops.drop(columns=["geometry", "visit_target", "place_name", "organizer",
                             "freq_type", "ss_importance"]),
         dummies],
        axis=1,
    ).reset_index(drop=True)

    report["n_final_shops"] = len(features)
    report["n_feature_columns"] = len(numeric_feature_columns(features))
    return features, report


def numeric_feature_columns(features: pd.DataFrame) -> list[str]:
    """Model-ready numeric columns (excludes identifiers, raw categoricals, target)."""
    excluded = set(ID_COLUMNS) | {TARGET_COLUMN, "has_target"}
    return [
        col
        for col in features.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(features[col])
    ]


def feature_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """NaN-free numeric matrix used by the model phases."""
    return features[numeric_feature_columns(features)].astype(np.float64)
