"""Smart Retail Intelligence dashboard.

Eight views over the pipeline artifacts: overview, in-depth dataset
description, method & pipeline (including the Hadoop/Spark distributed run),
one tab per research question structured as cause → method → result →
interpretation, and the full written report. Sidebar filters apply to every
analytical view.
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
from smart_retail.data import loader, schema
from smart_retail.features.engineering import TARGET_COLUMN, numeric_feature_columns

# Validated categorical palette (fixed slot order — clusters keep their color
# no matter what is filtered; worst adjacent CVD deltaE 24.2, see dataviz skill)
CLUSTER_COLORS = [
    "#2a78d6", "#1baf7a", "#eda100", "#008300",
    "#4a3aa7", "#e34948", "#e87ba4", "#eb6834",
]
SEQUENTIAL_HUE = "#2a78d6"
STATUS_SERIOUS = "#e34948"
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
    for key, path in {
        "metrics": CFG.reports_dir / pipeline.METRICS_JSON,
        "dataset_profile": CFG.reports_dir / "dataset_profile.json",
    }.items():
        if path.exists():
            art[key] = json.loads(path.read_text())
    # every Spark run: spark_metrics.json = national, spark_metrics_<divs>.json = regional
    spark_dir = CFG.data_dir / "spark"
    if spark_dir.exists():
        runs = {}
        for path in sorted(spark_dir.glob("spark_metrics*.json")):
            scope = path.stem.replace("spark_metrics", "").strip("_") or "national (all divisions)"
            runs[scope.replace("_", " + ")] = json.loads(path.read_text())
        if runs:
            art["spark_runs"] = runs
            art["spark_metrics"] = next(iter(runs.values()))
    for key, path in {
        "profiles": CFG.reports_dir / "cluster_profiles.csv",
        "shap": CFG.reports_dir / "shap_importance.csv",
    }.items():
        if path.exists():
            art[key] = pd.read_csv(path)
    try:
        art["products"] = loader.load_products(CFG)
    except (FileNotFoundError, ValueError):
        pass
    return art


@st.cache_data(show_spinner="Joining analysis layers…")
def build_base(_art_keys: tuple) -> pd.DataFrame:
    """One row per shop: features + cluster + risk + anomaly + prediction."""
    art = load_artifacts()
    base = art["features"]
    joins = {
        "clusters": ["shop_code", "cluster"],
        "risk": ["shop_code", "at_risk", "risk_score"],
        "anomalies": ["shop_code", "combined_score"],
        "predictions": ["shop_code", "predicted"],
    }
    for key, cols in joins.items():
        if key in art:
            base = base.merge(art[key][cols], on="shop_code", how="left")
    return base


@st.cache_data(ttl=30, show_spinner=False)
def cluster_topology() -> dict | None:
    """Live HDFS + Spark topology via their HTTP APIs (compose network)."""
    import requests

    try:
        jmx = requests.get(
            "http://namenode:9870/jmx?qry=Hadoop:service=NameNode,name=NameNodeInfo",
            timeout=2,
        ).json()["beans"][0]
        live_nodes = json.loads(jmx.get("LiveNodes", "{}"))
        master = requests.get("http://spark-master:8080/json/", timeout=2).json()
        return {
            "datanodes": [
                {"datanode": name.split(":")[0],
                 "used_gb": round(info.get("usedSpace", 0) / 1e9, 2),
                 "blocks": info.get("numBlocks", 0),
                 "state": info.get("adminState", "?")}
                for name, info in live_nodes.items()
            ],
            "workers": [
                {"worker": w["host"], "cores": w["cores"], "state": w["state"],
                 "memory_gb": round(w["memory"] / 1024, 1)}
                for w in master.get("workers", [])
            ],
            "spark_status": master.get("status", "?"),
        }
    except Exception:
        return None


def require(art: dict, *keys: str) -> bool:
    missing = [k for k in keys if k not in art]
    if missing:
        st.info(
            "This view needs pipeline artifacts that are not present yet "
            f"({', '.join(missing)}). Run: `docker compose run --rm pipeline`"
        )
        return False
    return True


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
    base_chart = alt.Chart(df).encode(
        x=alt.X(value, title=value_title or value),
        y=alt.Y(label, sort="-x", title=None),
        tooltip=list(df.columns),
    )
    return (
        base_chart.mark_bar(color=color, cornerRadiusEnd=4, height={"band": 0.6})
        + base_chart.mark_text(align="left", dx=4, color="#555").encode(
            text=alt.Text(value, format=fmt)
        )
    )


def histogram(df: pd.DataFrame, column: str, title: str, clip_q: float = 0.99) -> alt.Chart:
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


def section(title: str, body: str) -> None:
    """A labelled prose block used for the cause / interpretation passages."""
    st.markdown(f"**{title}**")
    st.markdown(body)


# --------------------------------------------------------------------------
# load + global filters
# --------------------------------------------------------------------------

art = load_artifacts()

st.markdown(
    """
<div style="padding:1.1rem 1.4rem;border-radius:10px;margin-bottom:0.8rem;
     background:linear-gradient(100deg,#f0f5fc 0%,#fafbfd 70%);
     border:1px solid #e3eaf3;">
  <div style="font-size:1.55rem;font-weight:700;color:#16385c;">
    Smart Retail Intelligence
  </div>
  <div style="color:#41556e;margin-top:0.15rem;">
    Bangladesh FMCG distribution network · 719,817 GPS-tagged outlets
  </div>
  <div style="margin-top:0.55rem;">
    <code style="background:#e8f0fa;border-radius:4px;padding:2px 8px;">MITE 431 · Big Data Analytics</code>
    <code style="background:#e8f0fa;border-radius:4px;padding:2px 8px;">Mendeley mgzvngzng2 · CC BY 4.0</code>
    <code style="background:#e8f0fa;border-radius:4px;padding:2px 8px;">Ratul Sikder · Fahim Rahman</code>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if "features" not in art:
    require(art, "features")
    st.stop()

base = build_base(tuple(sorted(art.keys())))

with st.container(border=True):
    fc1, fc2, fc3, fc4, fc5, fc6 = st.columns([2, 2, 2, 2, 1.6, 1.4])
    sel_divisions = fc1.multiselect(
        "Division", sorted(base["division"].unique()), placeholder="All divisions"
    )
    cluster_options = (
        sorted(base["cluster"].dropna().unique().astype(int))
        if "cluster" in base else []
    )
    sel_clusters = fc2.multiselect(
        "Archetype", cluster_options, placeholder="All archetypes",
        format_func=lambda c: f"Cluster {c}",
    )
    sel_types = fc3.multiselect(
        "Shop type", sorted(base["shop_type"].unique()), placeholder="All types"
    )
    sel_importance = fc4.multiselect(
        "Importance", sorted(base["shop_importance"].unique()), placeholder="All tiers"
    )
    with fc5:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        only_target = st.checkbox("Has sales target")

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
    fc6.metric("Shops in view", f"{len(f):,}",
               f"{len(f) / len(base):.1%} of {len(base):,}", delta_color="off")

if f.empty:
    st.warning("The current filter combination matches no shops — clear a filter.")
    st.stop()

(tab_overview, tab_dataset, tab_method, tab_rq1, tab_rq2,
 tab_rq3, tab_rq4, tab_report) = st.tabs(
    ["Overview", "Dataset", "Method & pipeline", "RQ1 · Archetypes",
     "RQ2 · Sales prediction", "RQ3 · Target-miss risk", "RQ4 · Anomalies", "Report"]
)

# ==========================================================================
with tab_overview:
    quality = art.get("metrics", {}).get("phase_1_preprocessing", {})

    st.subheader("Dataset at a glance")
    st.caption(
        "Full dataset — not affected by the sidebar filters. Source: Mendeley "
        "`mgzvngzng2` v1 (Shahin, Swarna & Farid, 2025), collected Feb 2022 – Oct 2024."
    )
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Raw shop records", f"{quality.get('n_raw_shop_rows', len(base)):,}")
    d2.metric(
        "Outlets analysed", f"{quality.get('n_final_shops', len(base)):,}",
        f"-{quality.get('n_invalid_or_missing_gps', 0):,} missing GPS", delta_color="off",
    )
    d3.metric(
        "Shops with sales target",
        f"{quality.get('n_shops_with_target', int(base['has_target'].sum())):,}",
        f"{quality.get('n_shops_with_target', 0) / max(quality.get('n_final_shops', 1), 1):.0%} of outlets",
        delta_color="off",
    )
    if "products" in art:
        products = art["products"]
        d4.metric(
            "SKU variants", len(products),
            f"{products['brand'].nunique()} brands · {products['type'].nunique()} types",
            delta_color="off",
        )
    else:
        d4.metric("SKU variants", "42", "5 brands · 9 types", delta_color="off")
    d5.metric("Targeted products / shop", len(schema.PRODUCT_TARGET_COLUMNS))

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Divisions", base["division"].nunique())
    g2.metric("Districts (zila)", base["zila"].nunique())
    g3.metric("Sub-districts (upazila)", base["upazila"].nunique())
    g4.metric("Model features", quality.get("n_feature_columns", "—"))
    st.divider()

    st.subheader("Current view")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Shops in view", f"{len(f):,}")
    c2.metric("Divisions", f["division"].nunique())
    c3.metric("Districts", f["zila"].nunique())
    c4.metric("With target", f"{int(f['has_target'].sum()):,}",
              f"{f['has_target'].mean():.0%}", delta_color="off")
    if "risk_score" in f:
        c5.metric("Mean predicted risk", f"{f['risk_score'].mean():.1%}")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Network map**")
        sample = f.sample(min(MAP_SAMPLE, len(f)), random_state=0)
        if "cluster" in sample:
            st.caption(f"{len(sample):,}-shop random sample, colored by archetype")
            st.markdown(cluster_legend(sample["cluster"].dropna().unique()),
                        unsafe_allow_html=True)
            color_fn = lambda row: cluster_color(row.get("cluster"))
        else:
            color_fn = lambda row: SEQUENTIAL_HUE
        st_folium(
            shop_map(sample, color_fn, ["shop_code", "division", "shop_type"]),
            height=460, use_container_width=True, returned_objects=[],
        )
    with right:
        st.markdown("**Shops per division**")
        counts = f["division"].value_counts().rename_axis("division").reset_index(name="shops")
        st.altair_chart(
            hbar(counts, "shops", "division", value_title="shops", fmt=",").properties(height=280),
            width="stretch",
        )
        st.markdown("**Shops per type (top 12)**")
        type_counts = (
            f["shop_type"].value_counts().head(12).rename_axis("shop_type")
            .reset_index(name="shops")
        )
        st.altair_chart(
            hbar(type_counts, "shops", "shop_type", value_title="shops", fmt=",").properties(height=280),
            width="stretch",
        )
        st.caption(
            "'Unknown' = shop_type was never captured in the source survey "
            "(10.23% of raw records) — kept as an explicit category rather "
            "than dropped. Full explanation on the Dataset tab."
        )

# ==========================================================================
def render_column_detail(record: dict) -> None:
    """Full page for one column: meaning, central tendency, distribution,
    and how the pipeline handles its gaps."""
    if record.get("description"):
        st.markdown(f"**What it means.** {record['description']}")

    total = record["non_null"] + record["missing"]
    if record["missing"] > 0:
        st.warning(
            f"**{record['missing']:,} missing values ({record['missing_pct']}% "
            f"of {total:,} rows).** {record.get('transformation') or 'Left as-is.'}",
            icon="⚠️",
        )
    elif record.get("transformation"):
        st.caption(f"Handling: {record['transformation']}")

    if record.get("kind") == "numeric":
        st.markdown("**Central tendency & spread**")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Mean", f"{record['mean']:,}")
        m2.metric("Median", f"{record['median']:,}")
        m3.metric("Mode", f"{record['mode']:,}" if record.get("mode") is not None else "—")
        m4.metric("Std dev", f"{record['std']:,}" if record.get("std") is not None else "—")
        m5.metric("Skew", record.get("skew", "—"))
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Min", f"{record['min']:,}")
        r2.metric("25th pct", f"{record['q25']:,}")
        r3.metric("75th pct", f"{record['q75']:,}")
        r4.metric("95th pct", f"{record['p95']:,}")
        r5.metric("Max", f"{record['max']:,}")
        if record.get("skew") is not None and record["skew"] > 2:
            st.caption(
                f"Skew {record['skew']} — heavily right-tailed: the mean "
                f"({record['mean']:,}) sits far above the median "
                f"({record['median']:,}), pulled up by a few extreme values. "
                "Use the median as the 'typical' value."
            )
        hist = record.get("histogram")
        if hist:
            bins = pd.DataFrame({
                "from": hist["edges"][:-1],
                "to": hist["edges"][1:],
                "rows": hist["counts"],
            })
            chart = (
                alt.Chart(bins)
                .mark_bar(color=SEQUENTIAL_HUE, opacity=0.85, binSpacing=1)
                .encode(
                    x=alt.X("from:Q", title=record["column"],
                            axis=alt.Axis(format="~s")),
                    x2="to:Q",
                    y=alt.Y("rows:Q", title="rows"),
                    tooltip=["from", "to", "rows"],
                )
                .properties(height=220, title="Distribution (clipped at the 99th percentile)")
            )
            st.altair_chart(chart, width="stretch")
    else:
        st.markdown("**Category landscape**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Distinct values", f"{record['unique']:,}")
        c2.metric("Most common", record.get("most_common", "—"),
                  f"{record.get('most_common_count', 0):,} rows", delta_color="off")
        c3.metric("Least common", record.get("least_common", "—"),
                  f"{record.get('least_common_count', 0):,} rows", delta_color="off")
        top = pd.DataFrame(record.get("top_values", []))
        if not top.empty:
            top["value"] = top["value"].replace("<missing>", "(missing → 'Unknown')")
            st.altair_chart(
                hbar(top, "count", "value", value_title="rows", fmt=",")
                .properties(height=48 + 26 * len(top),
                            title=f"Value distribution (top {len(top)})"),
                width="stretch",
            )


with tab_dataset:
    st.subheader("In-depth dataset description")
    profile = art.get("dataset_profile")
    if not profile:
        st.info(
            "Column-level profile not generated yet — re-run phase 1: "
            "`docker compose run --rm pipeline python -m smart_retail.pipeline --phases 1`"
        )
    else:
        ts = profile["target_summary"]
        cat = profile["catalogue_summary"]
        st.markdown(
            "Three interlinked files describe the network. Every column of "
            "every file is profiled below from the actual data — pick a file, "
            "then a column, to see its meaning, central tendency, "
            "distribution, and exactly how the pipeline treats its gaps."
        )
        st.info(
            "**What is the 'Unknown' shop type?** 85,167 shop records "
            "(10.23%) have *no* shop_type in the raw file — the survey never "
            "captured it. The same records mostly lack style, importance and "
            "organizer too. Phase 1 keeps these shops and labels the gap "
            "explicitly as **'Unknown'** instead of deleting 85k rows; the "
            "clustering then isolates them as their own archetype (the "
            "data-poor 'ghost-shop' segment of RQ1) — so the gap itself "
            "became a finding.",
            icon="❓",
        )

        file_tabs = st.tabs([f"📄 {name}" for name in profile["files"]])
        for file_tab, (filename, file_profile) in zip(file_tabs, profile["files"].items()):
            with file_tab:
                cols = file_profile["columns"]
                cols_df = pd.DataFrame(cols)
                n_numeric = int((cols_df.get("kind") == "numeric").sum())
                h1, h2, h3, h4 = st.columns(4)
                h1.metric("Rows", f"{file_profile['rows']:,}")
                h2.metric("Columns", len(cols))
                h3.metric("Numeric / categorical", f"{n_numeric} / {len(cols) - n_numeric}")
                h4.metric("Columns with gaps", int((cols_df["missing"] > 0).sum()))

                summary_cols = [c for c in ("column", "description", "dtype",
                                            "missing_pct", "unique", "examples")
                                if c in cols_df.columns]
                with st.expander("Column summary table", expanded=False):
                    st.dataframe(
                        cols_df[summary_cols], width="stretch", hide_index=True,
                        column_config={
                            "missing_pct": st.column_config.NumberColumn(
                                "missing %", format="%.2f%%"),
                        },
                    )

                selected = st.selectbox(
                    "Explore a column", [c["column"] for c in cols],
                    key=f"col_{filename}",
                )
                record = next(c for c in cols if c["column"] == selected)
                render_column_detail(record)

        st.divider()
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Target rows with a Total", f"{ts['rows_with_total']:,}")
        k2.metric("Target rows all-NaN", f"{ts['rows_all_nan']:,}")
        k3.metric("Duplicate shop codes (targets)", f"{ts['duplicate_shop_codes']:,}")
        k4.metric("Median monthly Total", f"{ts['total_median']:,.2f}")
        k5.metric("95th percentile Total", f"{ts['total_p95']:,.2f}")
        st.caption(
            f"The Total target is heavily right-skewed: median {ts['total_median']:,.2f} "
            f"vs mean {ts['total_mean']:,.2f} vs max {ts['total_max']:,.0f} — a few "
            "wholesale-style outlets carry targets thousands of times the typical shop."
        )
        if "products" in art:
            st.markdown(
                f"**Product catalogue** — {cat['skus']} SKUs, {cat['brands']} brands, "
                f"{cat['product_types']} flavour/variant types, {cat['base_products']} base products"
            )
            st.dataframe(art["products"], width="stretch", hide_index=True, height=300)

# ==========================================================================
with tab_method:
    st.subheader("How the analysis is produced")
    st.graphviz_chart("""
digraph pipeline {
  rankdir=LR; bgcolor=transparent;
  node [shape=box, style="rounded,filled", fillcolor="#eef4fb", color="#2a78d6",
        fontname="Helvetica", fontsize=11];
  edge [color="#8a8a8a", fontname="Helvetica", fontsize=9];
  raw [label="Raw CSVs\\n832,881 shops · 42 SKUs"];
  hdfs [label="HDFS\\nnamenode + datanode", fillcolor="#fdf3e2", color="#eda100"];
  spark [label="Spark cluster\\ndistributed ETL + MLlib", fillcolor="#fdf3e2", color="#eda100"];
  p1 [label="Phase 1\\ncleaning + features"];
  p2 [label="Phase 2 (RQ1)\\nclustering"];
  p3 [label="Phase 3 (RQ2)\\nregression"];
  p4 [label="Phase 4 (RQ3)\\nrisk classification"];
  p5 [label="Phase 5 (RQ4)\\nanomaly detection"];
  dash [label="Dashboard + report", fillcolor="#e8f5ee", color="#1baf7a"];
  raw -> p1 -> p2;
  p2 -> p3; p2 -> p4; p2 -> p5;
  p3 -> dash; p4 -> dash; p5 -> dash;
  raw -> hdfs -> spark;
  spark -> dash [label="distributed run"];
}
""")
    st.caption(
        "Two execution paths over the same data: the scikit-learn pipeline "
        "(reference implementation, exact algorithms from the proposal) and "
        "the Hadoop/Spark path (distributed storage + MLlib) for the big-data "
        "requirement. Both start from the same raw CSVs."
    )
    st.divider()

    if require(art, "metrics"):
        quality = art["metrics"].get("phase_1_preprocessing", {})
        st.subheader("Data preparation, step by step")
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
            .encode(x=alt.X("shops", title="shops"),
                    y=alt.Y("stage", sort=None, title=None),
                    tooltip=["stage", "shops"])
        )
        labels = (
            alt.Chart(funnel).mark_text(align="left", dx=4, color="#555")
            .encode(x="shops", y=alt.Y("stage", sort=None),
                    text=alt.Text("shops", format=","))
        )
        st.altair_chart((bars + labels).properties(height=210), width="stretch")
        st.markdown(
            f"The {quality.get('n_invalid_or_missing_gps', 0):,} rows removed at "
            "stage 3 have empty latitude/longitude in the raw file "
            "(`geometry = POINT EMPTY`) — the missing-GPS rate the dataset "
            "authors themselves report. What remains equals the paper's "
            "published outlet count exactly, so no valid shop is lost."
        )
        st.divider()

        st.subheader("Methods, phase by phase")
        st.markdown(
            "Each phase consumes the previous phase's output. The table gives "
            "the one-line view; the notes underneath explain how each method "
            "works and why it was chosen."
        )
        methods = pd.DataFrame(
            [
                {"phase": "1 · Preparation", "input": "raw CSVs",
                 "method": "dedup, GPS validation, BallTree neighbour densities, district-centroid distance, one-hot encoding",
                 "output": "features.parquet (719,817 × 48)"},
                {"phase": "2 · Archetypes (RQ1)", "input": "scaled feature matrix",
                 "method": "K-Means scanned k=3…9, scored by silhouette + Davies-Bouldin; hierarchical & DBSCAN as cross-checks",
                 "output": "cluster per shop, profiles"},
                {"phase": "3 · Sales (RQ2)", "input": "features of 613,660 labelled shops",
                 "method": "linear baseline vs Random Forest, XGBoost, LightGBM on an 80/20 split; SHAP on the winner",
                 "output": "predictions, feature attributions"},
                {"phase": "4 · Risk (RQ3)", "input": "features + cluster",
                 "method": "peer-relative miss label; SMOTE inside the training fold; LogReg / RF / XGBoost",
                 "output": "risk score per shop"},
                {"phase": "5 · Anomalies (RQ4)", "input": "features + cluster",
                 "method": "Isolation Forest + LOF fitted per cluster, combined by percentile rank",
                 "output": "ranked top decile for triage"},
            ]
        )
        st.dataframe(methods, width="stretch", hide_index=True)

        with st.expander("Notes on the methods (read before interpreting results)"):
            st.markdown(
                """
The **density features** count how many other shops sit within 500 m, 1 km and
5 km of each outlet. A BallTree with the haversine metric answers all 719,817
queries in seconds and, unlike a naive pairwise scan, scales as *n log n*.
Density is the pipeline's proxy for local competition and urbanisation.

The **cluster count k = 8** is not assumed — K-Means is fitted for every k from
3 to 9 and judged on two independent criteria. Silhouette rewards tight,
well-separated clusters; Davies–Bouldin penalises overlapping ones. Both point
to 8. DBSCAN collapsing to one cluster with 98.7% noise is reported rather than
hidden: it says the retail fabric is continuous, which is itself a finding.

For **sales prediction** the linear baseline exists to prove non-linearity: it
manages R² 0.02 where XGBoost reaches 0.45 with identical inputs. SHAP values
are computed on the winner because gradient-boosted trees are otherwise opaque;
they decompose every single prediction into per-feature contributions.

The **risk label is a proxy** and is labelled as such everywhere: the dataset
carries assigned targets, not realized sales, so a shop is flagged when its
target sits far below what shops of the same archetype and type receive. SMOTE
runs strictly inside the training fold — resampling before the split would leak
synthetic copies of test shops into training and inflate every metric.

**Anomaly scores are computed within each cluster**, never globally: a rural
kiosk compared against metro superstores would always look anomalous. Isolation
Forest catches shops that are easy to isolate in feature space; LOF catches
shops whose local neighbourhood density differs from their neighbours'. The
final rank is the mean of the two percentile ranks.
"""
            )
        st.divider()

        st.subheader("Distributed run — Hadoop HDFS + Spark MLlib")

        topology = cluster_topology()
        if topology:
            t1, t2, t3 = st.columns([1, 2, 2])
            t1.metric("Spark master", topology["spark_status"])
            with t2:
                st.markdown("**Live datanodes (HDFS)**")
                st.dataframe(pd.DataFrame(topology["datanodes"]),
                             width="stretch", hide_index=True)
            with t3:
                st.markdown("**Live Spark workers**")
                st.dataframe(pd.DataFrame(topology["workers"]),
                             width="stretch", hide_index=True)
            st.caption(
                "Queried live from the namenode and Spark master APIs. Scale "
                "with `WORKERS=N DATANODES=N make spark-up` — blocks (stored "
                "with replication 2) spread across whatever datanodes exist."
            )
        else:
            st.caption(
                "Cluster offline — start it with `make spark-up` to see the "
                "live topology here."
            )

        spark_runs = art.get("spark_runs")
        if not spark_runs:
            st.info(
                "No distributed-run metrics yet. Start the cluster and run: "
                "`make spark-up && make spark-pipeline` — results appear here."
            )
        else:
            scope = st.selectbox(
                "Run scope (regional runs read only their divisions' HDFS "
                "partitions — partition pruning)",
                list(spark_runs.keys()),
            )
            spark_metrics = spark_runs[scope]
            run_scope = spark_metrics.get("run_scope", {})
            st.markdown(
                f"Storage: `{spark_metrics.get('storage', '')}` · input: "
                f"{run_scope.get('input_source', 'raw CSVs')} · scope: "
                f"**{run_scope.get('divisions', 'all')}**"
            )
            rows = []
            for key in ("phase_1_spark", "phase_2_spark", "phase_3_spark",
                        "phase_4_spark", "phase_5_spark"):
                p = spark_metrics.get(key)
                if p:
                    rows.append({
                        "phase": key.replace("_spark", "").replace("_", " "),
                        "engine": p.get("engine", ""),
                        "duration (s)": p.get("duration_sec"),
                        "headline": p.get("best_model")
                        or (f"best k = {p['best_k']}" if "best_k" in p else "")
                        or (f"{p.get('n_top_decile', ''):,} top-decile" if p.get("n_top_decile") else ""),
                    })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            if len(spark_runs) > 1:
                st.markdown("**All Spark runs, side by side**")
                comparison = []
                for name, m in spark_runs.items():
                    p1, p2 = m.get("phase_1_spark", {}), m.get("phase_2_spark", {})
                    p3, p4 = m.get("phase_3_spark", {}), m.get("phase_4_spark", {})
                    comparison.append({
                        "run": name,
                        "shops": p1.get("n_final_shops"),
                        "best k": p2.get("best_k"),
                        "regression R² (GBT)": round(
                            p3.get("models", {}).get("gbt", {}).get("r2", float("nan")), 3),
                        "risk ROC-AUC (GBT)": round(
                            p4.get("models", {}).get("gbt", {}).get("roc_auc", float("nan")), 3),
                        "total (s)": m.get("total_duration_sec"),
                    })
                st.dataframe(pd.DataFrame(comparison), width="stretch", hide_index=True)

            with st.expander("Full distributed-run metrics (JSON)"):
                st.json(spark_metrics)
            st.caption(
                "MLlib offers no XGBoost/SHAP/SMOTE/Isolation-Forest, so the "
                "distributed run uses the nearest equivalents (GBT, class "
                "weights, centroid distance). The scikit-learn pipeline stays "
                "the reference for the exact proposal algorithms; matching row "
                "counts across both paths cross-validate the data preparation."
            )
        st.divider()

        st.subheader("Engineered feature attributes")
        feature_cols = numeric_feature_columns(art["features"])
        onehot_families: dict[str, list] = {}
        core = []
        for col in feature_cols:
            family = next(
                (p for p in ("shop_type_", "shop_style_", "shop_importance_",
                             "place_nature_", "division_") if col.startswith(p)), None,
            )
            if family:
                onehot_families.setdefault(family.rstrip("_"), []).append(col)
            else:
                core.append(col)
        descriptions = {
            "density_500m": "Shops within 500 m (BallTree haversine, self excluded)",
            "density_1km": "Shops within 1 km",
            "density_5km": "Shops within 5 km",
            "dist_to_district_centre_km": "Distance to the district's shop centroid",
            "n_products_targeted": "Products (of 17) with a positive monthly target",
            "latitude": "Validated GPS latitude",
            "longitude": "Validated GPS longitude",
        }
        dictionary = pd.DataFrame(
            [{"feature": c, "group": "spatial" if c.startswith(("density", "dist"))
              else "core", "description": descriptions.get(c, "")} for c in core]
            + [{"feature": f"{fam}_* ({len(cols)} columns)", "group": "one-hot",
                "description": f"Indicator per {fam.replace('_', ' ')} level; 'Unknown' kept explicit"}
               for fam, cols in onehot_families.items()]
        )
        c1, c2 = st.columns([2, 3])
        with c1:
            st.metric("Model features", len(feature_cols))
            st.dataframe(dictionary, width="stretch", hide_index=True, height=360)
        with c2:
            st.caption("Distributions in the current filter view (clipped at p99)")
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

# ==========================================================================
with tab_rq1:
    if require(art, "profiles", "metrics") and "cluster" in f:
        clustering = art["metrics"].get("phase_2_clustering", {})
        best_k = clustering.get("best_k", "?")

        section(
            "Why this question",
            "Planning across 720,000 outlets treats a Dhaka wholesale hub and a "
            "rural kiosk the same way. If natural shop archetypes exist, targets, "
            "logistics and field visits can be planned per archetype instead of "
            "one-size-fits-all.",
        )
        with st.expander("Method"):
            st.markdown(
                "All 48 features are standardized and K-Means is fitted for every "
                "k from 3 to 9 (10 restarts each, seed 42). Each k is scored with "
                "silhouette (cluster separation, higher better) and Davies–Bouldin "
                "(cluster overlap, lower better). Agglomerative clustering and "
                "DBSCAN run on a 20k subsample as structural cross-checks."
            )

        st.subheader(f"Result — {best_k} archetypes")
        km = clustering.get("kmeans", {})
        if km:
            selection = pd.DataFrame(
                [{"k": int(k), "silhouette": v["silhouette"],
                  "davies_bouldin": v["davies_bouldin"]} for k, v in km.items()]
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
                    .encode(x=alt.X("k:O", title="k"), y=alt.Y(metric, title=metric),
                            tooltip=["k", metric])
                    .properties(height=210, title=f"{metric} ({better})")
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
                color=alt.Color("archetype", legend=None,
                                scale=alt.Scale(domain=counts["archetype"].tolist(),
                                                range=[cluster_color(c) for c in counts["cluster"]])),
                tooltip=["archetype", "shops"],
            )
            .properties(height=60 + 30 * len(counts), title="Cluster sizes (current view)")
        )
        labels = (
            alt.Chart(counts).mark_text(align="left", dx=4, color="#555")
            .encode(x="shops", y=alt.Y("archetype", sort=None),
                    text=alt.Text("shops", format=","))
        )
        st.altair_chart(bars + labels, width="stretch")
        st.markdown("**Cluster profiles (medians, full dataset)**")
        st.dataframe(art["profiles"], width="stretch", hide_index=True)

        profiles = art["profiles"]
        key_account = profiles.loc[profiles[TARGET_COLUMN].idxmax()]
        ghost = profiles[profiles["n_products_targeted"] == 0]
        interpretation = (
            f"The 8 archetypes map cleanly onto geography and function. Cluster "
            f"{int(key_account['cluster'])} is the key-account tier — median target "
            f"{key_account[TARGET_COLUMN]:,.0f}, several times every other cluster. "
        )
        if not ghost.empty:
            g = ghost.iloc[0]
            interpretation += (
                f"Cluster {int(g['cluster'])} ({g['share']:.0%} of shops) has zero "
                "targeted products and unknown type — a machine-found candidate "
                "list for the ghost-shop audit. "
            )
        interpretation += (
            "DBSCAN finding one big cluster (98.7% noise at default settings) "
            "says the network is a continuous fabric rather than isolated "
            "islands — archetypes overlap at the edges, which the silhouette "
            "of ~0.19 also reflects."
        )
        section("Interpretation", interpretation)

# ==========================================================================
with tab_rq2:
    if require(art, "predictions", "metrics"):
        reg = art["metrics"].get("phase_3_regression", {})
        section(
            "Why this question",
            "If a shop's expected monthly sales can be predicted from where it is "
            "and what kind of shop it is, targets can be set from evidence — and "
            "large gaps between model and book become a signal in themselves.",
        )
        with st.expander("Method"):
            st.markdown(
                "The 613,660 shops with a target are split 80/20. A linear "
                "regression baseline is benchmarked against Random Forest, XGBoost "
                "and LightGBM on identical features; the winner (lowest test RMSE) "
                "is explained with SHAP, which decomposes each prediction into "
                "per-feature contributions."
            )

        st.subheader(f"Result — best model: {reg.get('best_model', '?')}")
        models = pd.DataFrame(reg.get("models", {})).T.reset_index(names="model")
        st.dataframe(
            models.style.format({"rmse": "{:.3f}", "mae": "{:.3f}", "r2": "{:.3f}"}),
            width="stretch", hide_index=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(
                hbar(models[["model", "rmse"]], "rmse", "model").properties(
                    height=220, title="Test RMSE (lower is better)"),
                width="stretch",
            )
        with c2:
            if "shap" in art:
                st.altair_chart(
                    hbar(art["shap"].head(15), "mean_abs_shap", "feature",
                         value_title="mean |SHAP|").properties(
                        height=340, title="Top 15 feature attributions"),
                    width="stretch",
                )

        preds = f[f["has_target"] & f["predicted"].notna()]
        if not preds.empty:
            st.markdown("**Actual vs predicted (current view, sampled)**")
            preds = preds.sample(min(5_000, len(preds)), random_state=0)
            limit = float(preds[[TARGET_COLUMN, "predicted"]].quantile(0.99).max())
            scatter = (
                alt.Chart(preds.rename(columns={TARGET_COLUMN: "actual"}))
                .mark_circle(size=14, color=SEQUENTIAL_HUE, opacity=0.35)
                .encode(x=alt.X("actual", scale=alt.Scale(domain=(0, limit), clamp=True)),
                        y=alt.Y("predicted", scale=alt.Scale(domain=(0, limit), clamp=True)),
                        tooltip=["shop_code", "division", "actual", "predicted"])
            )
            diagonal = (
                alt.Chart(pd.DataFrame({"v": [0, limit]}))
                .mark_line(color="#999", strokeDash=[4, 4]).encode(x="v", y="v")
            )
            st.altair_chart((scatter + diagonal).properties(height=360), width="stretch")

        baseline_r2 = reg.get("models", {}).get("linear_regression", {}).get("r2", float("nan"))
        best_r2 = reg.get("models", {}).get(reg.get("best_model", ""), {}).get("r2", float("nan"))
        section(
            "Interpretation",
            f"The linear baseline explains almost nothing (R² {baseline_r2:.2f}) while "
            f"{reg.get('best_model', '?')} reaches R² {best_r2:.2f} on the same inputs — the "
            "relationship is real but non-linear. SHAP puts assortment breadth and "
            "importance tier first, then micro-location (density within 500 m, "
            "coordinates, distance to the district centre). Shops far above the "
            "diagonal in the scatter are under-targeted relative to their profile; "
            "far below, over-targeted.",
        )

# ==========================================================================
with tab_rq3:
    if require(art, "metrics") and "risk_score" in f:
        clf = art["metrics"].get("phase_4_classification", {})
        section(
            "Why this question",
            "Target shortfalls are currently discovered at quarter-end, after the "
            "quarter is lost. A classifier that flags at-risk shops in advance "
            "turns that from an autopsy into a to-do list.",
        )
        with st.expander("Method"):
            st.markdown(
                f"The dataset has assigned targets, not realized sales, so the label "
                f"is an explicit proxy: `{clf.get('label_definition', '')}` "
                f"(threshold {clf.get('risk_threshold', 0.8)}). Base rate "
                f"{clf.get('label_rate', 0):.1%}. Logistic regression, Random Forest "
                "and XGBoost are trained on a stratified 80/20 split; SMOTE "
                "oversampling runs only inside the training fold so the test set "
                "is never touched."
            )

        st.subheader(f"Result — best model: {clf.get('best_model', '?')}")
        models = pd.DataFrame(clf.get("models", {})).T.reset_index(names="model")
        st.dataframe(
            models.style.format("{:.3f}", subset=models.columns[1:].tolist()),
            width="stretch", hide_index=True,
        )
        risk_view = f[f["risk_score"].notna()]
        if not risk_view.empty:
            c1, c2 = st.columns(2)
            with c1:
                by_division = (
                    risk_view.groupby("division")["risk_score"].mean()
                    .rename("mean risk").rename_axis("division").reset_index()
                )
                st.altair_chart(
                    hbar(by_division, "mean risk", "division", color=STATUS_SERIOUS,
                         fmt=".1%").properties(height=270, title="Mean predicted risk by division"),
                    width="stretch",
                )
            with c2:
                if "cluster" in risk_view:
                    by_cluster = (
                        risk_view.groupby("cluster")["risk_score"].mean()
                        .rename("mean risk").rename_axis("cluster").reset_index()
                    )
                    by_cluster["archetype"] = by_cluster["cluster"].map(lambda c: f"Cluster {int(c)}")
                    st.altair_chart(
                        hbar(by_cluster[["archetype", "mean risk"]], "mean risk", "archetype",
                             color=STATUS_SERIOUS, fmt=".1%").properties(
                            height=270, title="Mean predicted risk by archetype"),
                        width="stretch",
                    )
            st.markdown("**Highest-risk shops (current view)**")
            st.dataframe(
                risk_view.sort_values("risk_score", ascending=False)
                [["shop_code", "division", "zila", "shop_type", "shop_importance",
                  "cluster", TARGET_COLUMN, "risk_score", "at_risk"]].head(200),
                width="stretch", hide_index=True,
            )

        best_auc = clf.get("models", {}).get(clf.get("best_model", ""), {}).get("roc_auc", float("nan"))
        risk_all = base[base["risk_score"].notna()] if "risk_score" in base else pd.DataFrame()
        geo_note = ""
        if not risk_all.empty:
            by_div_all = risk_all.groupby("division")["at_risk"].mean().drop("Unknown", errors="ignore").sort_values()
            geo_note = (
                f" Geographically, predicted risk runs from {by_div_all.iloc[0]:.0%} in "
                f"{by_div_all.index[0]} to {by_div_all.iloc[-1]:.0%} in {by_div_all.index[-1]} — "
                "a ready-made priority order for field teams."
            )
        section(
            "Interpretation",
            f"ROC-AUC {best_auc:.2f} from structural features alone is strong "
            "discrimination for a proxy label; with realized sales the identical "
            "pipeline retrains into a true will-miss-next-quarter model." + geo_note,
        )

# ==========================================================================
with tab_rq4:
    if require(art, "metrics") and "combined_score" in f:
        anom = art["metrics"].get("phase_5_anomaly", {})
        section(
            "Why this question",
            "An unknown share of the distributor's book is duplicates, ghost shops "
            "or mis-entered data, and a few legitimate shops behave so unlike their "
            "peers that they signal either fraud or untapped demand. Ranking every "
            "shop by peer-relative strangeness turns a 720k-row audit into a "
            "prioritised worklist.",
        )
        with st.expander("Method"):
            st.markdown(
                "Isolation Forest and Local Outlier Factor are fitted separately "
                "within each of the 8 archetypes, so every shop is compared only "
                "with its own kind. Each method's score becomes a percentile rank "
                "within the cluster; the final score is their mean. The top decile "
                "is exported for manual triage."
            )

        anomaly_view = f[f["combined_score"].notna()]
        c1, c2, c3 = st.columns(3)
        c1.metric("Shops scored (view)", f"{len(anomaly_view):,}")
        c2.metric("Top decile in view", f"{len(anomaly_view[anomaly_view['combined_score'] >= 0.90]):,}")
        c3.metric("Top decile (full run)", f"{anom.get('n_top_anomalies', 0):,}")

        st.subheader("Result — ranked anomalies")
        map_top = (
            anomaly_view.sort_values("combined_score", ascending=False)
            .dropna(subset=["latitude", "longitude"]).head(MAP_SAMPLE)
        )
        if not map_top.empty:
            st_folium(
                shop_map(map_top, lambda row: STATUS_SERIOUS,
                         ["shop_code", "division", "shop_type", "combined_score"]),
                height=400, use_container_width=True, returned_objects=[],
            )
            st.dataframe(
                map_top[["shop_code", "cluster", "combined_score", "division", "zila",
                         "shop_type", "shop_importance", "density_1km", TARGET_COLUMN]].head(500),
                width="stretch", hide_index=True,
            )
        section(
            "Interpretation",
            "A high combined score means both detectors agree the shop is unlike "
            "its archetype peers. Field triage sorts these into three buckets: "
            "data errors (wrong GPS, duplicate entry), fraud candidates (targets "
            "or importance inconsistent with surroundings), and hidden "
            "opportunities (thriving shops in thin markets). Shops that are "
            "simultaneously in the RQ3 high-risk list deserve the first visits.",
        )

# ==========================================================================
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
        st.subheader("Analysis report — dataset to conclusions")
        st.caption(
            "Generated live from reports/metrics.json (full dataset, not filtered), "
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
