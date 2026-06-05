"""Smart Retail Intelligence dashboard.

Reads the artifacts produced by `python -m smart_retail.pipeline` and renders
seven views: network overview, pipeline & feature engineering, shop archetypes
(RQ1), sales prediction (RQ2), target-miss risk (RQ3), anomalies (RQ4), and a
full start-to-end analysis report. Sidebar filters (division / archetype /
shop type / target coverage) apply to every view.
"""

from __future__ import annotations

import json

import altair as alt
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from smart_retail import pipeline, reporting
from smart_retail.config import Config
from smart_retail.features.engineering import TARGET_COLUMN, numeric_feature_columns

# Validated categorical palette (fixed slot order — clusters keep their color
# no matter what is filtered; worst adjacent CVD deltaE 24.2, see dataviz skill)
CLUSTER_COLORS = [
    "#2a78d6", "#1baf7a", "#eda100", "#008300",
    "#4a3aa7", "#e34948", "#e87ba4", "#eb6834",
]
SEQUENTIAL_HUE = "#2a78d6"   # magnitude bars: one hue, not one-per-bar
STATUS_SERIOUS = "#e34948"   # reserved for risk/anomaly marks only
MAP_SAMPLE = 3_000

st.set_page_config(page_title="Smart Retail Intelligence", layout="wide")

CFG = Config.from_env()


# --------------------------------------------------------------------------
# data access
# --------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading pipeline artifacts…")
def load_artifacts() -> dict:
    art: dict = {}
    processed = {
        "features": pipeline.FEATURES_PARQUET,
        "clusters": pipeline.CLUSTERS_PARQUET,
        "predictions": pipeline.PREDICTIONS_PARQUET,
        "risk": pipeline.RISK_PARQUET,
        "anomalies": pipeline.ANOMALIES_PARQUET,
        "top_anomalies": pipeline.TOP_ANOMALIES_PARQUET,
    }
    for key, filename in processed.items():
        path = CFG.processed_dir / filename
        if path.exists():
            art[key] = pd.read_parquet(path)
    metrics_path = CFG.reports_dir / pipeline.METRICS_JSON
    if metrics_path.exists():
        art["metrics"] = json.loads(metrics_path.read_text())
    profiles_path = CFG.reports_dir / "cluster_profiles.csv"
    if profiles_path.exists():
        art["profiles"] = pd.read_csv(profiles_path)
    shap_path = CFG.reports_dir / "shap_importance.csv"
    if shap_path.exists():
        art["shap"] = pd.read_csv(shap_path)
    return art


@st.cache_data(show_spinner="Joining analysis layers…")
def build_base(_art_keys: tuple) -> pd.DataFrame:
    """One row per shop: features + cluster + risk + anomaly score."""
    art = load_artifacts()
    base = art["features"]
    if "clusters" in art:
        base = base.merge(art["clusters"], on="shop_code", how="left")
    if "risk" in art:
        base = base.merge(
            art["risk"][["shop_code", "at_risk", "risk_score"]],
            on="shop_code", how="left",
        )
    if "anomalies" in art:
        base = base.merge(
            art["anomalies"][["shop_code", "combined_score"]],
            on="shop_code", how="left",
        )
    if "predictions" in art:
        base = base.merge(
            art["predictions"][["shop_code", "predicted"]],
            on="shop_code", how="left",
        )
    return base


def require(art: dict, *keys: str) -> bool:
    missing = [k for k in keys if k not in art]
    if missing:
        st.info(
            "No pipeline artifacts found yet for this view "
            f"(missing: {', '.join(missing)}). Run:\n\n"
            "```\ndocker compose run --rm pipeline\n```"
        )
        return False
    return True


# --------------------------------------------------------------------------
# algorithm walkthrough cards (phase → input / internal steps / output)
# --------------------------------------------------------------------------

ALGORITHM_CARDS = [
    {
        "phase": "Phase 1",
        "title": "Preprocessing & feature engineering",
        "algorithm": "BallTree (haversine) · vectorized great-circle geometry",
        "input": "`shop_info_with_geo.csv` (832,881 rows) + "
                 "`product_target_for_shop.csv` (853,732 rows)",
        "steps": [
            "Drop duplicate `shop_code` rows (keep first occurrence).",
            "Validate GPS: latitude/longitude must exist and fall inside the "
            "Bangladesh bounding box (20.5–26.7°N, 88.0–92.7°E); "
            "`POINT EMPTY` rows are removed.",
            "Fill missing categoricals with an explicit `Unknown` level "
            "instead of dropping rows.",
            "Build a BallTree on radian coordinates and count neighbours "
            "within 500 m / 1 km / 5 km (haversine metric, self excluded).",
            "Compute each shop's great-circle distance to the centroid of its "
            "district's shops.",
            "Join monthly targets (valued rows preferred over NaN duplicates); "
            "derive `n_products_targeted` and `has_target`.",
            "One-hot encode shop type, style, importance, place nature, "
            "division → 48 numeric features.",
        ],
        "output": "`features.parquet` — 719,817 shops × 48 model features "
                  "+ identifiers and target",
    },
    {
        "phase": "Phase 2 · RQ1",
        "title": "Archetype clustering",
        "algorithm": "K-Means (selected) · Agglomerative + DBSCAN cross-checks",
        "input": "Standard-scaled 48-feature matrix (719,817 × 48)",
        "steps": [
            "Standardize every feature (zero mean, unit variance).",
            "Fit K-Means for k = 3…9 with 10 restarts each (seed 42).",
            "Score every k: silhouette on a 10k subsample (higher better) and "
            "Davies–Bouldin on the full matrix (lower better).",
            "Select the k with the best silhouette → k = 8.",
            "Cross-check structure: Agglomerative (Ward) on a 20k subsample, "
            "DBSCAN (eps 0.5, min 50) on the same subsample.",
            "Profile each cluster: median density, distance, target, size, "
            "dominant division/type/importance.",
        ],
        "output": "`clusters.parquet` (shop → cluster id) · "
                  "`cluster_profiles.csv` · `clustering_kmeans.joblib`",
    },
    {
        "phase": "Phase 3 · RQ2",
        "title": "Sales-target regression",
        "algorithm": "Linear baseline vs Random Forest · XGBoost · LightGBM + SHAP",
        "input": "Features of the 613,660 shops with a target; "
                 "label y = `total_target`",
        "steps": [
            "Split 80/20 train/test (seed 42).",
            "Fit four models: LinearRegression (scaled pipeline), "
            "RandomForest (100 trees, depth ≤ 20), XGBoost (300 × depth 8), "
            "LightGBM (300 trees, 63 leaves).",
            "Evaluate each on the untouched test set: RMSE, MAE, R².",
            "Pick the lowest-RMSE model → XGBoost.",
            "Explain it with SHAP TreeExplainer on ≤ 2,000 test rows; "
            "rank features by mean |SHAP value|.",
            "Predict all labelled shops with the winner.",
        ],
        "output": "`predictions.parquet` (actual vs predicted per shop) · "
                  "`shap_importance.csv` · `regression_best.joblib`",
    },
    {
        "phase": "Phase 4 · RQ3",
        "title": "Target-miss risk classification",
        "algorithm": "SMOTE + Logistic Regression · Random Forest · XGBoost",
        "input": "Features + cluster id of labelled shops; proxy label "
                 "`at_risk = total_target < 0.8 × median(total_target | cluster, shop_type)`",
        "steps": [
            "Build the peer-relative label per (cluster × shop type) group "
            "→ 36.6% base rate.",
            "Stratified 80/20 split (seed 42).",
            "Wrap every model in an imblearn pipeline `[SMOTE → model]` so "
            "minority oversampling happens ONLY inside fit() — the test set "
            "is never resampled.",
            "Fit LogisticRegression (scaled), RandomForest, XGBoost.",
            "Score on the raw test set: ROC-AUC, F1, precision, recall.",
            "Pick the highest ROC-AUC → XGBoost; score every labelled shop "
            "with predict_proba → risk_score ∈ [0, 1].",
        ],
        "output": "`risk.parquet` (at_risk label + risk_score per shop) · "
                  "`classification_best.joblib`",
    },
    {
        "phase": "Phase 5 · RQ4",
        "title": "Per-cluster anomaly detection",
        "algorithm": "Isolation Forest + Local Outlier Factor, combined by rank",
        "input": "Features + cluster labels for all 719,817 shops",
        "steps": [
            "For each cluster with ≥ 20 shops (smaller ones are skipped):",
            "— standardize the cluster's feature block;",
            "— Isolation Forest (200 trees): negate `score_samples` so higher "
            "= more anomalous, convert to percentile rank within the cluster;",
            "— LOF (20 neighbours): same negate-and-rank treatment;",
            "Combine: `combined_score` = mean of the two percentile ranks.",
            "Sort descending and export the top decile with full context "
            "(division, type, density, target) for manual triage into "
            "fraud / data error / hidden opportunity.",
        ],
        "output": "`anomalies.parquet` (scores for all shops) · "
                  "`top_anomalies.parquet` (71,982 ranked shops)",
    },
]


def render_algorithm_card(card: dict) -> None:
    with st.container(border=True):
        st.markdown(f"**{card['phase']} — {card['title']}**")
        st.caption(card["algorithm"])
        st.markdown(f"📥 **Input:** {card['input']}")
        with st.expander("Internal steps", expanded=False):
            for i, step in enumerate(card["steps"], start=1):
                st.markdown(f"{i}. {step}")
        st.markdown(f"📤 **Output:** {card['output']}")


# --------------------------------------------------------------------------
# chart helpers
# --------------------------------------------------------------------------

def cluster_color(cluster_id) -> str:
    if pd.isna(cluster_id):
        return "#999999"
    return CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)]


def cluster_legend(cluster_ids) -> str:
    chips = " ".join(
        f'<span style="display:inline-block;width:0.8em;height:0.8em;'
        f'border-radius:2px;background:{cluster_color(c)};"></span>'
        f'&nbsp;Cluster {int(c)}&nbsp;&nbsp;'
        for c in sorted(cluster_ids)
    )
    return f'<div style="line-height:1.8">{chips}</div>'


def hbar(df: pd.DataFrame, value: str, label: str, color=SEQUENTIAL_HUE,
         value_title: str | None = None, fmt: str = ",.3f") -> alt.Chart:
    """Horizontal magnitude bars: one hue, sorted, direct value labels."""
    base = alt.Chart(df).encode(
        x=alt.X(value, title=value_title or value),
        y=alt.Y(label, sort="-x", title=None),
        tooltip=list(df.columns),
    )
    return (
        base.mark_bar(color=color, cornerRadiusEnd=4, height={"band": 0.6})
        + base.mark_text(align="left", dx=4, color="#555").encode(
            text=alt.Text(value, format=fmt)
        )
    )


def histogram(df: pd.DataFrame, column: str, title: str, clip_q: float = 0.99) -> alt.Chart:
    """Distribution of one engineered feature, clipped at the 99th percentile."""
    values = df[[column]].dropna()
    cap = float(values[column].quantile(clip_q))
    clipped = values[values[column] <= cap]
    return (
        alt.Chart(clipped)
        .mark_bar(color=SEQUENTIAL_HUE, opacity=0.85, binSpacing=1)
        .encode(
            x=alt.X(column, bin=alt.Bin(maxbins=40), title=title),
            y=alt.Y("count()", title="shops"),
            tooltip=[alt.Tooltip(column, bin=True), "count()"],
        )
        .properties(height=180)
    )


def shop_map(df: pd.DataFrame, color_fn, popup_cols: list[str]) -> folium.Map:
    m = folium.Map(
        location=[df["latitude"].mean(), df["longitude"].mean()],
        zoom_start=7, tiles="cartodbpositron",
    )
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3, weight=0,
            fill=True, fill_color=color_fn(row), fill_opacity=0.7,
            popup=" | ".join(f"{c}: {row[c]}" for c in popup_cols if c in row),
        ).add_to(m)
    return m


# --------------------------------------------------------------------------
# load + global filters
# --------------------------------------------------------------------------

art = load_artifacts()

st.title("Smart Retail Intelligence — Bangladesh FMCG Network")
st.caption(
    "719,817 GPS-tagged outlets · Mendeley dataset mgzvngzng2 (CC BY 4.0) · "
    "MITE 431, Ratul Sikder & Fahim Rahman"
)

if "features" not in art:
    require(art, "features")
    st.stop()

base = build_base(tuple(sorted(art.keys())))

with st.sidebar:
    st.header("Filters")
    st.caption("Empty selection = everything. Filters apply to every tab.")

    sel_divisions = st.multiselect(
        "Division", sorted(base["division"].unique()),
    )
    cluster_options = (
        sorted(base["cluster"].dropna().unique().astype(int))
        if "cluster" in base else []
    )
    sel_clusters = st.multiselect(
        "Archetype (cluster)", cluster_options,
        format_func=lambda c: f"Cluster {c}",
    )
    sel_types = st.multiselect(
        "Shop type", sorted(base["shop_type"].unique()),
    )
    sel_importance = st.multiselect(
        "Shop importance", sorted(base["shop_importance"].unique()),
    )
    only_target = st.checkbox("Only shops with a sales target", value=False)

    mask = pd.Series(True, index=base.index)
    if sel_divisions:
        mask &= base["division"].isin(sel_divisions)
    if sel_clusters and "cluster" in base:
        mask &= base["cluster"].isin(sel_clusters)
    if sel_types:
        mask &= base["shop_type"].isin(sel_types)
    if sel_importance:
        mask &= base["shop_importance"].isin(sel_importance)
    if only_target:
        mask &= base["has_target"]

    f = base[mask]
    st.divider()
    st.metric(
        "Shops in view", f"{len(f):,}",
        f"{len(f) / len(base):.1%} of {len(base):,}",
    )

if f.empty:
    st.warning("The current filter combination matches no shops — clear a filter.")
    st.stop()

(tab_overview, tab_pipeline, tab_clusters, tab_sales,
 tab_risk, tab_anomaly, tab_report) = st.tabs(
    ["Overview", "Pipeline & features", "Shop archetypes", "Sales prediction",
     "Target-miss risk", "Anomalies", "Full report"]
)

# --------------------------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Shops in view", f"{len(f):,}")
    c2.metric("Divisions", f["division"].nunique())
    c3.metric("Districts (zila)", f["zila"].nunique())
    c4.metric(
        "With sales target",
        f"{int(f['has_target'].sum()):,}",
        f"{f['has_target'].mean():.0%} coverage",
    )
    if "risk_score" in f:
        c5.metric("Mean predicted risk", f"{f['risk_score'].mean():.1%}")
    st.divider()

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Network map")
        sample = f.sample(min(MAP_SAMPLE, len(f)), random_state=0)
        if "cluster" in sample:
            st.caption(f"{len(sample):,}-shop random sample of the current view, colored by archetype")
            st.markdown(
                cluster_legend(sample["cluster"].dropna().unique()),
                unsafe_allow_html=True,
            )
            color_fn = lambda row: cluster_color(row.get("cluster"))
        else:
            st.caption(f"{len(sample):,}-shop random sample of the current view")
            color_fn = lambda row: SEQUENTIAL_HUE
        st_folium(
            shop_map(sample, color_fn, ["shop_code", "division", "shop_type"]),
            height=480, use_container_width=True, returned_objects=[],
        )
    with right:
        st.subheader("Shops per division")
        counts = (
            f["division"].value_counts().rename_axis("division")
            .reset_index(name="shops")
        )
        st.altair_chart(
            hbar(counts, "shops", "division", value_title="shops", fmt=",")
            .properties(height=300),
            width="stretch",
        )
        st.subheader("Shops per type")
        type_counts = (
            f["shop_type"].value_counts().head(12).rename_axis("shop_type")
            .reset_index(name="shops")
        )
        st.altair_chart(
            hbar(type_counts, "shops", "shop_type", value_title="shops", fmt=",")
            .properties(height=300),
            width="stretch",
        )

# --------------------------------------------------------------------------
with tab_pipeline:
    if require(art, "metrics"):
        quality = art["metrics"].get("phase_1_preprocessing", {})
        st.subheader("Preprocessing funnel (phase 1)")
        st.caption(
            "Every stage of the cleaning pipeline, from the raw CSV to the "
            "model-ready shop table — full dataset, not affected by filters."
        )
        funnel = pd.DataFrame(
            [
                {"stage": "1 · Raw shop records", "shops": quality.get("n_raw_shop_rows", 0)},
                {"stage": "2 · Unique shop codes", "shops": quality.get("n_unique_shops", 0)},
                {"stage": "3 · Valid GPS (inside Bangladesh)", "shops": quality.get("n_final_shops", 0)},
                {"stage": "4 · Joined with a sales target", "shops": quality.get("n_shops_with_target", 0)},
            ]
        )
        bars = (
            alt.Chart(funnel)
            .mark_bar(color=SEQUENTIAL_HUE, cornerRadiusEnd=4, height={"band": 0.55})
            .encode(
                x=alt.X("shops", title="shops"),
                y=alt.Y("stage", sort=None, title=None),
                tooltip=["stage", "shops"],
            )
        )
        labels = (
            alt.Chart(funnel)
            .mark_text(align="left", dx=4, color="#555")
            .encode(x="shops", y=alt.Y("stage", sort=None), text=alt.Text("shops", format=","))
        )
        st.altair_chart((bars + labels).properties(height=220), width="stretch")

        dropped = quality.get("n_invalid_or_missing_gps", 0)
        st.markdown(
            f"**Why {dropped:,} rows drop at stage 3.** Their latitude/longitude "
            "cells are empty in the raw CSV (`geometry = POINT EMPTY`) — the "
            "13.6% missing-GPS rate documented by the dataset authors. After "
            "removal the count equals the paper's published 719,817 exactly; "
            "zero valid coordinates fall outside Bangladesh."
        )
        st.divider()

        st.subheader("Algorithm walkthrough — every step, input → output")
        st.caption(
            "What each phase consumes, what it computes internally, and what "
            "it hands to the next phase. Expand a card for the in-depth steps."
        )
        row1 = st.columns(2)
        row2 = st.columns(2)
        row3 = st.columns(2)
        slots = [row1[0], row1[1], row2[0], row2[1], row3[0]]
        for slot, card in zip(slots, ALGORITHM_CARDS):
            with slot:
                render_algorithm_card(card)
        st.divider()

        st.subheader("Engineered feature attributes")
        feature_cols = numeric_feature_columns(art["features"])
        onehot_families = {}
        core = []
        for col in feature_cols:
            family = next(
                (p for p in ("shop_type_", "shop_style_", "shop_importance_",
                             "place_nature_", "division_") if col.startswith(p)),
                None,
            )
            if family:
                onehot_families.setdefault(family.rstrip("_"), []).append(col)
            else:
                core.append(col)

        descriptions = {
            "density_500m": "Shops within 500 m (BallTree haversine, self excluded)",
            "density_1km": "Shops within 1 km",
            "density_5km": "Shops within 5 km",
            "dist_to_district_centre_km": "Great-circle distance to the district's shop centroid",
            "n_products_targeted": "Products (of 17) with a positive monthly target",
            "latitude": "Validated GPS latitude",
            "longitude": "Validated GPS longitude",
        }
        dictionary = pd.DataFrame(
            [
                {"feature": c, "group": "engineered / spatial"
                 if c.startswith(("density", "dist")) else "core",
                 "description": descriptions.get(c, "")}
                for c in core
            ]
            + [
                {"feature": f"{family}_* ({len(cols)} columns)",
                 "group": "one-hot structural",
                 "description": f"Indicator per {family.replace('_', ' ')} level, "
                                "'Unknown' kept as an explicit level"}
                for family, cols in onehot_families.items()
            ]
        )
        c1, c2 = st.columns([2, 3])
        with c1:
            st.metric("Model features", len(feature_cols))
            st.dataframe(dictionary, width="stretch", hide_index=True, height=380)
        with c2:
            st.caption("Distributions in the current filter view (clipped at the 99th percentile)")
            hist_sample = f.sample(min(50_000, len(f)), random_state=0)
            g1, g2 = st.columns(2)
            g1.altair_chart(histogram(hist_sample, "density_1km", "shops within 1 km"), width="stretch")
            g2.altair_chart(histogram(hist_sample, "dist_to_district_centre_km", "km to district centre"), width="stretch")
            g3, g4 = st.columns(2)
            g3.altair_chart(histogram(hist_sample, "n_products_targeted", "products targeted"), width="stretch")
            if f["has_target"].any():
                g4.altair_chart(
                    histogram(hist_sample[hist_sample["has_target"]], TARGET_COLUMN, "monthly total target"),
                    width="stretch",
                )

        st.subheader("Categorical completeness")
        st.caption("Share of shops with an 'Unknown' (originally missing) value, current view")
        unknown_share = pd.DataFrame(
            [
                {"attribute": col, "unknown share": float((f[col] == "Unknown").mean())}
                for col in ("shop_type", "shop_style", "shop_importance", "place_nature", "division")
            ]
        )
        st.altair_chart(
            hbar(unknown_share, "unknown share", "attribute", fmt=".1%")
            .properties(height=200),
            width="stretch",
        )

# --------------------------------------------------------------------------
with tab_clusters:
    if require(art, "profiles", "metrics") and "cluster" in f:
        clustering = art["metrics"].get("phase_2_clustering", {})
        st.subheader(f"K-Means archetypes — best k = {clustering.get('best_k', '?')}")

        km = clustering.get("kmeans", {})
        if km:
            selection = pd.DataFrame(
                [
                    {"k": int(k), "silhouette": v["silhouette"],
                     "davies_bouldin": v["davies_bouldin"]}
                    for k, v in km.items()
                ]
            ).sort_values("k")
            c1, c2 = st.columns(2)
            for col, metric, better in (
                (c1, "silhouette", "higher is better"),
                (c2, "davies_bouldin", "lower is better"),
            ):
                line = (
                    alt.Chart(selection)
                    .mark_line(color=SEQUENTIAL_HUE, strokeWidth=2,
                               point=alt.OverlayMarkDef(size=80, color=SEQUENTIAL_HUE))
                    .encode(
                        x=alt.X("k:O", title="k"),
                        y=alt.Y(metric, title=metric),
                        tooltip=["k", metric],
                    )
                    .properties(height=220, title=f"{metric} ({better})")
                )
                col.altair_chart(line, width="stretch")

        counts = (
            f["cluster"].value_counts().rename_axis("cluster")
            .reset_index(name="shops").sort_values("cluster")
        )
        counts["archetype"] = counts["cluster"].map(lambda c: f"Cluster {int(c)}")
        bars = (
            alt.Chart(counts)
            .mark_bar(cornerRadiusEnd=4, height={"band": 0.6})
            .encode(
                x=alt.X("shops", title="shops"),
                y=alt.Y("archetype", sort=None, title=None),
                color=alt.Color(
                    "archetype",
                    scale=alt.Scale(
                        domain=counts["archetype"].tolist(),
                        range=[cluster_color(c) for c in counts["cluster"]],
                    ),
                    legend=None,
                ),
                tooltip=["archetype", "shops"],
            )
            .properties(height=60 + 30 * len(counts), title="Cluster sizes (current view)")
        )
        labels = (
            alt.Chart(counts)
            .mark_text(align="left", dx=4, color="#555")
            .encode(x="shops", y=alt.Y("archetype", sort=None),
                    text=alt.Text("shops", format=","))
        )
        st.altair_chart(bars + labels, width="stretch")

        st.subheader("Cluster profiles (medians, full dataset)")
        st.dataframe(art["profiles"], width="stretch", hide_index=True)

        st.caption(
            "Cross-checks — agglomerative silhouette: "
            f"{clustering.get('agglomerative', {}).get('silhouette', float('nan')):.3f} · "
            f"DBSCAN found {clustering.get('dbscan', {}).get('n_clusters_found', '?')} clusters "
            f"({clustering.get('dbscan', {}).get('noise_share', 0):.0%} noise)"
        )

# --------------------------------------------------------------------------
with tab_sales:
    if require(art, "predictions", "metrics"):
        reg = art["metrics"].get("phase_3_regression", {})
        st.subheader(f"Model benchmark — best: {reg.get('best_model', '?')}")
        st.caption("Benchmark metrics come from the held-out test set of the full training run.")
        models = pd.DataFrame(reg.get("models", {})).T.reset_index(names="model")
        st.dataframe(
            models.style.format({"rmse": "{:.3f}", "mae": "{:.3f}", "r2": "{:.3f}"}),
            width="stretch", hide_index=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(
                hbar(models[["model", "rmse"]], "rmse", "model").properties(
                    height=220, title="Test RMSE (lower is better)"
                ),
                width="stretch",
            )
        with c2:
            if "shap" in art:
                top_shap = art["shap"].head(15)
                st.altair_chart(
                    hbar(top_shap, "mean_abs_shap", "feature",
                         value_title="mean |SHAP|").properties(
                        height=350, title="Top 15 feature attributions"
                    ),
                    width="stretch",
                )

        st.subheader("Actual vs predicted (current view, sampled)")
        preds = f[f["has_target"] & f["predicted"].notna()]
        if preds.empty:
            st.info("No shops with predictions match the current filters.")
        else:
            preds = preds.sample(min(5_000, len(preds)), random_state=0)
            limit = float(preds[[TARGET_COLUMN, "predicted"]].quantile(0.99).max())
            scatter = (
                alt.Chart(preds.rename(columns={TARGET_COLUMN: "actual"}))
                .mark_circle(size=14, color=SEQUENTIAL_HUE, opacity=0.35)
                .encode(
                    x=alt.X("actual", scale=alt.Scale(domain=(0, limit), clamp=True)),
                    y=alt.Y("predicted", scale=alt.Scale(domain=(0, limit), clamp=True)),
                    tooltip=["shop_code", "division", "actual", "predicted"],
                )
            )
            diagonal = (
                alt.Chart(pd.DataFrame({"v": [0, limit]}))
                .mark_line(color="#999", strokeDash=[4, 4])
                .encode(x="v", y="v")
            )
            st.altair_chart((scatter + diagonal).properties(height=380), width="stretch")

# --------------------------------------------------------------------------
with tab_risk:
    if require(art, "metrics") and "risk_score" in f:
        clf = art["metrics"].get("phase_4_classification", {})
        st.subheader(f"Target-miss risk — best: {clf.get('best_model', '?')}")
        st.caption(f"Label (proxy): `{clf.get('label_definition', '')}` · "
                   f"base rate {clf.get('label_rate', 0):.1%}")

        models = pd.DataFrame(clf.get("models", {})).T.reset_index(names="model")
        st.dataframe(
            models.style.format("{:.3f}", subset=models.columns[1:].tolist()),
            width="stretch", hide_index=True,
        )

        risk_view = f[f["risk_score"].notna()]
        if risk_view.empty:
            st.info("No scored shops match the current filters.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                by_division = (
                    risk_view.groupby("division")["risk_score"].mean()
                    .rename("mean risk").rename_axis("division").reset_index()
                )
                st.altair_chart(
                    hbar(by_division, "mean risk", "division",
                         color=STATUS_SERIOUS, fmt=".1%").properties(
                        height=280, title="Mean predicted risk by division (current view)"
                    ),
                    width="stretch",
                )
            with c2:
                if "cluster" in risk_view:
                    by_cluster = (
                        risk_view.groupby("cluster")["risk_score"].mean()
                        .rename("mean risk").rename_axis("cluster").reset_index()
                    )
                    by_cluster["archetype"] = by_cluster["cluster"].map(
                        lambda c: f"Cluster {int(c)}"
                    )
                    st.altair_chart(
                        hbar(by_cluster[["archetype", "mean risk"]], "mean risk",
                             "archetype", color=STATUS_SERIOUS, fmt=".1%").properties(
                            height=280, title="Mean predicted risk by archetype (current view)"
                        ),
                        width="stretch",
                    )

            st.subheader("Highest-risk shops (current view)")
            st.dataframe(
                risk_view.sort_values("risk_score", ascending=False)
                [["shop_code", "division", "zila", "shop_type", "shop_importance",
                  "cluster", TARGET_COLUMN, "risk_score", "at_risk"]].head(200),
                width="stretch", hide_index=True,
            )

# --------------------------------------------------------------------------
with tab_anomaly:
    if require(art, "metrics") and "combined_score" in f:
        anom = art["metrics"].get("phase_5_anomaly", {})
        anomaly_view = f[f["combined_score"].notna()]
        c1, c2, c3 = st.columns(3)
        c1.metric("Shops scored (view)", f"{len(anomaly_view):,}")
        top_decile = anomaly_view[anomaly_view["combined_score"] >= 0.90]
        c2.metric("Top-decile anomalies (view)", f"{len(top_decile):,}")
        c3.metric("Skipped small clusters (full run)", anom.get("n_skipped_small_clusters", 0))

        st.subheader("Anomaly map (highest combined score in view)")
        st.caption(
            "Isolation Forest + LOF fitted within each cluster; triage as "
            "fraud / data error / hidden opportunity"
        )
        map_top = (
            anomaly_view.sort_values("combined_score", ascending=False)
            .dropna(subset=["latitude", "longitude"]).head(MAP_SAMPLE)
        )
        if map_top.empty:
            st.info("No scored shops match the current filters.")
        else:
            st_folium(
                shop_map(map_top, lambda row: STATUS_SERIOUS,
                         ["shop_code", "division", "shop_type", "combined_score"]),
                height=420, use_container_width=True, returned_objects=[],
            )
            st.subheader("Ranked anomalies (current view)")
            st.dataframe(
                map_top[["shop_code", "cluster", "combined_score", "division",
                         "zila", "shop_type", "shop_importance", "density_1km",
                         TARGET_COLUMN]].head(500),
                width="stretch", hide_index=True,
            )

# --------------------------------------------------------------------------
with tab_report:
    if require(art, "metrics"):
        risk_by_division = None
        if "risk_score" in base.columns:
            risk_by_division = base.groupby("division")["at_risk"].mean()
        sections = reporting.build_report(
            metrics=art["metrics"],
            profiles=art.get("profiles"),
            shap_importance=art.get("shap"),
            risk_by_division=risk_by_division,
            top_anomalies=art.get("top_anomalies"),
        )

        st.subheader("Analysis report — dataset to conclusions, end to end")
        st.caption(
            "Generated live from reports/metrics.json (full dataset, not filtered) "
            "so the numbers always match the latest pipeline run."
        )
        st.download_button(
            "Download report (markdown)",
            data=reporting.report_markdown(sections),
            file_name="smart_retail_analysis_report.md",
            mime="text/markdown",
        )
        for section_md in sections.values():
            st.markdown(section_md)
            st.divider()

        durations = art["metrics"].get("durations_sec", {})
        if durations:
            st.caption(
                "Pipeline durations — "
                + " · ".join(f"{k.replace('_', ' ')}: {v:.0f}s" for k, v in durations.items())
                + f" · total {sum(durations.values()):.0f}s"
            )
