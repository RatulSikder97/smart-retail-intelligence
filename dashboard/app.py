"""Smart Retail Intelligence dashboard.

Reads the artifacts produced by `python -m smart_retail.pipeline` and renders
five views: network overview, shop archetypes (RQ1), sales prediction (RQ2),
target-miss risk (RQ3), and anomalies (RQ4).
"""

from __future__ import annotations

import json

import altair as alt
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from smart_retail import pipeline
from smart_retail.config import Config

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


@st.cache_data(show_spinner=False)
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


def cluster_color(cluster_id: int) -> str:
    return CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)]


def hbar(df: pd.DataFrame, value: str, label: str, color=SEQUENTIAL_HUE,
         value_title: str | None = None) -> alt.Chart:
    """Horizontal magnitude bars: one hue, sorted, direct value labels."""
    base = alt.Chart(df).encode(
        x=alt.X(value, title=value_title or value),
        y=alt.Y(label, sort="-x", title=None),
        tooltip=list(df.columns),
    )
    return (
        base.mark_bar(color=color, cornerRadiusEnd=4, height={"band": 0.6})
        + base.mark_text(align="left", dx=4, color="#555").encode(
            text=alt.Text(value, format=",.3f")
        )
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


art = load_artifacts()

st.title("Smart Retail Intelligence — Bangladesh FMCG Network")
st.caption(
    "719,817 GPS-tagged outlets · Mendeley dataset mgzvngzng2 (CC BY 4.0) · "
    "MITE 431, Ratul Sikder & Fahim Rahman"
)

tab_overview, tab_clusters, tab_sales, tab_risk, tab_anomaly = st.tabs(
    ["Overview", "Shop archetypes", "Sales prediction", "Target-miss risk", "Anomalies"]
)

with tab_overview:
    if require(art, "features", "metrics"):
        features, metrics = art["features"], art["metrics"]
        quality = metrics.get("phase_1_preprocessing", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Shops analysed", f"{len(features):,}")
        c2.metric("Divisions", features["division"].nunique())
        c3.metric("Districts (zila)", features["zila"].nunique())
        c4.metric(
            "Shops with sales target",
            f"{int(features['has_target'].sum()):,}",
            f"{features['has_target'].mean():.0%} coverage",
        )
        st.divider()

        left, right = st.columns([3, 2])
        with left:
            st.subheader("Network map")
            sample = features.sample(min(MAP_SAMPLE, len(features)), random_state=0)
            if "clusters" in art:
                sample = sample.merge(art["clusters"], on="shop_code", how="left")
                color_fn = lambda row: cluster_color(row.get("cluster", 0))
                st.caption(
                    f"{len(sample):,}-shop random sample, colored by cluster"
                )
            else:
                color_fn = lambda row: SEQUENTIAL_HUE
                st.caption(f"{len(sample):,}-shop random sample")
            st_folium(
                shop_map(sample, color_fn, ["shop_code", "division", "shop_type"]),
                height=480, use_container_width=True, returned_objects=[],
            )
        with right:
            st.subheader("Shops per division")
            counts = (
                features["division"].value_counts().rename_axis("division")
                .reset_index(name="shops")
            )
            st.altair_chart(
                hbar(counts, "shops", "division", value_title="shops").properties(height=300),
                width='stretch',
            )
            if quality:
                st.subheader("Data quality (phase 1)")
                st.json(quality)

with tab_clusters:
    if require(art, "clusters", "profiles", "metrics"):
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
                    .mark_line(color=SEQUENTIAL_HUE, strokeWidth=2, point=alt.OverlayMarkDef(size=80, color=SEQUENTIAL_HUE))
                    .encode(
                        x=alt.X("k:O", title="k"),
                        y=alt.Y(metric, title=metric),
                        tooltip=["k", metric],
                    )
                    .properties(height=220, title=f"{metric} ({better})")
                )
                col.altair_chart(line, width='stretch')

        counts = (
            art["clusters"]["cluster"].value_counts().rename_axis("cluster")
            .reset_index(name="shops").sort_values("cluster")
        )
        counts["archetype"] = counts["cluster"].map(lambda c: f"Cluster {c}")
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
            .properties(height=60 + 30 * len(counts), title="Cluster sizes")
        )
        labels = (
            alt.Chart(counts)
            .mark_text(align="left", dx=4, color="#555")
            .encode(
                x="shops", y=alt.Y("archetype", sort=None), text=alt.Text("shops", format=",")
            )
        )
        st.altair_chart(bars + labels, width='stretch')

        st.subheader("Cluster profiles (medians)")
        st.dataframe(art["profiles"], width='stretch', hide_index=True)

        st.caption(
            "Cross-checks — agglomerative silhouette: "
            f"{clustering.get('agglomerative', {}).get('silhouette', float('nan')):.3f} · "
            f"DBSCAN found {clustering.get('dbscan', {}).get('n_clusters_found', '?')} clusters "
            f"({clustering.get('dbscan', {}).get('noise_share', 0):.0%} noise)"
        )

with tab_sales:
    if require(art, "predictions", "metrics"):
        reg = art["metrics"].get("phase_3_regression", {})
        st.subheader(f"Model benchmark — best: {reg.get('best_model', '?')}")
        models = pd.DataFrame(reg.get("models", {})).T.reset_index(names="model")
        st.dataframe(
            models.style.format({"rmse": "{:.3f}", "mae": "{:.3f}", "r2": "{:.3f}"}),
            width='stretch', hide_index=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(
                hbar(models[["model", "rmse"]], "rmse", "model").properties(
                    height=220, title="Test RMSE (lower is better)"
                ),
                width='stretch',
            )
        with c2:
            if "shap" in art:
                top_shap = art["shap"].head(15)
                st.altair_chart(
                    hbar(
                        top_shap, "mean_abs_shap", "feature",
                        value_title="mean |SHAP|",
                    ).properties(height=350, title="Top 15 feature attributions"),
                    width='stretch',
                )

        st.subheader("Actual vs predicted (test-style scatter, sampled)")
        preds = art["predictions"].sample(
            min(5_000, len(art["predictions"])), random_state=0
        )
        limit = float(preds[["actual", "predicted"]].quantile(0.99).max())
        scatter = (
            alt.Chart(preds)
            .mark_circle(size=14, color=SEQUENTIAL_HUE, opacity=0.35)
            .encode(
                x=alt.X("actual", scale=alt.Scale(domain=(0, limit), clamp=True)),
                y=alt.Y("predicted", scale=alt.Scale(domain=(0, limit), clamp=True)),
                tooltip=["shop_code", "actual", "predicted"],
            )
        )
        diagonal = (
            alt.Chart(pd.DataFrame({"v": [0, limit]}))
            .mark_line(color="#999", strokeDash=[4, 4])
            .encode(x="v", y="v")
        )
        st.altair_chart(
            (scatter + diagonal).properties(height=380), width='stretch'
        )

with tab_risk:
    if require(art, "risk", "features", "metrics"):
        clf = art["metrics"].get("phase_4_classification", {})
        st.subheader(f"Target-miss risk — best: {clf.get('best_model', '?')}")
        st.caption(f"Label (proxy): `{clf.get('label_definition', '')}` · "
                   f"base rate {clf.get('label_rate', 0):.1%}")

        models = pd.DataFrame(clf.get("models", {})).T.reset_index(names="model")
        st.dataframe(
            models.style.format("{:.3f}", subset=models.columns[1:].tolist()),
            width='stretch', hide_index=True,
        )

        risk = art["risk"].merge(
            art["features"][["shop_code", "division", "zila", "shop_type"]],
            on="shop_code", how="left",
        )
        by_division = (
            risk.groupby("division")["risk_score"].mean().rename("mean risk")
            .rename_axis("division").reset_index()
        )
        st.altair_chart(
            hbar(by_division, "mean risk", "division", color=STATUS_SERIOUS).properties(
                height=280, title="Mean predicted risk by division"
            ),
            width='stretch',
        )

        st.subheader("Highest-risk shops")
        st.dataframe(
            risk.sort_values("risk_score", ascending=False).head(200),
            width='stretch', hide_index=True,
        )

with tab_anomaly:
    if require(art, "top_anomalies", "metrics"):
        anom = art["metrics"].get("phase_5_anomaly", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Shops scored", f"{anom.get('n_scored', 0):,}")
        c2.metric("Top-decile anomalies", f"{anom.get('n_top_anomalies', 0):,}")
        c3.metric("Small clusters skipped", anom.get("n_skipped_small_clusters", 0))

        top = art["top_anomalies"]
        st.subheader("Anomaly map (top of the ranking)")
        st.caption(
            "Isolation Forest + LOF fitted within each cluster; triage as "
            "fraud / data error / hidden opportunity"
        )
        map_top = top.dropna(subset=["latitude", "longitude"]).head(MAP_SAMPLE)
        st_folium(
            shop_map(
                map_top,
                lambda row: STATUS_SERIOUS,
                ["shop_code", "division", "shop_type", "combined_score"],
            ),
            height=420, width='stretch', returned_objects=[],
        )
        st.subheader("Ranked anomalies")
        st.dataframe(top.head(500), width='stretch', hide_index=True)
