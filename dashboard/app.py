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
from smart_retail.preprocessing import loader, schema
from smart_retail.preprocessing.features import TARGET_COLUMN, numeric_feature_columns

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
# design system — "research instrument": IBM Plex, hairlines, quiet chrome
# --------------------------------------------------------------------------

INK, MUTED, HAIRLINE, PANEL, NAVY = "#1C2A3A", "#5B6B7C", "#E5EAF0", "#F7F9FB", "#16385C"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] * { font-family: 'IBM Plex Sans', sans-serif; }
/* restore Streamlit's icon font (the override above would garble icons) */
[data-testid="stIconMaterial"], span[class*="material-symbols"] {
  font-family: 'Material Symbols Rounded' !important;
}
code, kbd, pre, [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }

.block-container { max-width: 1260px; padding-top: 0.8rem; padding-bottom: 3rem; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }

h1, h2, h3 { color: #16385C; letter-spacing: -0.01em; }

/* tabs as quiet mono nav */
button[data-baseweb="tab"] p {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.06em;
  color: #5B6B7C;
}
button[data-baseweb="tab"][aria-selected="true"] p { color: #1C2A3A; font-weight: 600; }
div[data-baseweb="tab-highlight"] { background-color: #2A78D6; height: 2px; }
div[data-baseweb="tab-border"] { background-color: #E5EAF0; }

/* dataframe headers in mono */
[data-testid="stDataFrame"] thead th, [data-testid="stDataFrame"] [role="columnheader"] {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.68rem !important; text-transform: uppercase; letter-spacing: 0.05em;
  color: #5B6B7C !important;
}

/* bordered containers: subtler radius + hairline */
div[data-testid="stVerticalBlockBorderWrapper"] > div {
  border-color: #E5EAF0 !important; border-radius: 8px;
}

/* metric fallback styling where st.metric remains */
[data-testid="stMetricLabel"] p {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.68rem !important; text-transform: uppercase;
  letter-spacing: 0.05em; color: #5B6B7C !important;
}
[data-testid="stMetricValue"] { font-size: 1.65rem; font-weight: 600; color: #1C2A3A; }
[data-testid="stMetricDelta"] { font-size: 0.72rem; }

.stat-row { display: grid; gap: 12px; margin: 0.35rem 0 0.9rem 0; }
.stat-tile {
  border: 1px solid #E5EAF0; border-radius: 8px; background: #fff;
  padding: 0.65rem 0.85rem;
}
.stat-tile.accent { border-left: 3px solid #2A78D6; }
.stat-label {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem;
  text-transform: uppercase; letter-spacing: 0.06em; color: #5B6B7C;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.stat-value {
  font-size: 1.55rem; font-weight: 600; color: #1C2A3A;
  font-variant-numeric: tabular-nums; line-height: 1.25; margin-top: 0.1rem;
}
.stat-context { font-size: 0.72rem; color: #5B6B7C; margin-top: 0.05rem; }

.eyebrow {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
  text-transform: uppercase; letter-spacing: 0.09em; color: #2A78D6;
  margin-bottom: 0.1rem;
}
.section-title { font-size: 1.22rem; font-weight: 650; color: #16385C; margin: 0; }
.section-desc { font-size: 0.86rem; color: #5B6B7C; margin: 0.15rem 0 0.7rem 0; }

.hdr {
  display: flex; justify-content: space-between; align-items: center;
  padding: 0.35rem 0 0.9rem 0;
}
.hdr-left { display: flex; align-items: center; gap: 0.8rem; }
.hdr-logo {
  width: 40px; height: 40px; border-radius: 9px; flex: none;
  background: #16385C; display: flex; align-items: flex-end;
  justify-content: center; gap: 3px; padding-bottom: 9px;
}
.hdr-logo i { display: block; width: 5px; border-radius: 2px; background: #7FB1E8; }
.hdr-logo i:nth-child(1) { height: 9px; }
.hdr-logo i:nth-child(2) { height: 16px; background: #fff; }
.hdr-logo i:nth-child(3) { height: 12px; }
.wordmark { font-size: 1.32rem; font-weight: 700; color: #16385C;
            letter-spacing: -0.015em; line-height: 1.15; }
.hdr-sub { font-size: 0.82rem; color: #5B6B7C; margin-top: 1px; }
.hdr-chips { display: flex; gap: 6px; flex: none; }
.hdr-chip {
  font-size: 0.7rem; color: #41556E; background: #F1F5F9;
  border: 1px solid #E5EAF0; border-radius: 999px; padding: 3px 11px;
  white-space: nowrap;
}

.readout {
  display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
  margin: 0 0 1.05rem 0;
}
.readout-tag {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.64rem;
  text-transform: uppercase; letter-spacing: 0.07em; color: #5B6B7C;
  margin-right: 2px;
}
.readout-pill {
  display: inline-flex; align-items: baseline; gap: 5px;
  border: 1px solid #E5EAF0; border-radius: 6px; background: #fff;
  padding: 3px 10px; font-size: 0.74rem; color: #5B6B7C;
}
.readout-pill b { color: #1C2A3A; font-weight: 600; font-variant-numeric: tabular-nums; }
</style>
""",
    unsafe_allow_html=True,
)


def _plex_theme():
    axis = {"labelColor": MUTED, "titleColor": MUTED, "labelFont": "IBM Plex Sans",
            "titleFont": "IBM Plex Sans", "labelFontSize": 11, "titleFontSize": 11,
            "gridColor": "#EEF2F6", "domain": False, "tickColor": "#EEF2F6",
            "labelLimit": 180}
    return {
        "config": {
            "font": "IBM Plex Sans",
            "axis": axis,
            # never drop category labels on y; keep smart thinning on x
            "axisY": {"labelOverlap": False},
            "view": {"stroke": None},
            "title": {"anchor": "start", "fontSize": 13, "fontWeight": 600,
                      "color": INK, "font": "IBM Plex Sans"},
            "legend": {"labelColor": MUTED, "titleColor": MUTED},
        }
    }


alt.themes.register("plex", _plex_theme)
alt.themes.enable("plex")


def stat_row(tiles: list[tuple], accent: bool = False) -> None:
    """Evenly boxed stat tiles: (label, value, context-or-None) triples."""
    cells = "".join(
        f'<div class="stat-tile{" accent" if accent else ""}">'
        f'<div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div>'
        + (f'<div class="stat-context">{context}</div>' if context else "")
        + "</div>"
        for label, value, context in tiles
    )
    st.markdown(
        f'<div class="stat-row" style="grid-template-columns: repeat({len(tiles)}, 1fr);">{cells}</div>',
        unsafe_allow_html=True,
    )


def section_header(eyebrow: str, title: str, description: str = "") -> None:
    st.markdown(
        f'<div class="eyebrow">{eyebrow}</div>'
        f'<p class="section-title">{title}</p>'
        + (f'<p class="section-desc">{description}</p>' if description else ""),
        unsafe_allow_html=True,
    )


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
        "imputation_sensitivity": CFG.reports_dir / "imputation_sensitivity.json",
        "alternatives": CFG.reports_dir / "alternatives.json",
        "imbalance": CFG.reports_dir / "imbalance.json",
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
    chips = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:4px;'
        f'font-size:0.74rem;color:{MUTED};margin-right:12px;">'
        f'<span style="display:inline-block;width:9px;height:9px;'
        f'border-radius:2px;background:{cluster_color(c)};"></span>'
        f'Cluster {int(c)}</span>'
        for c in sorted(cluster_ids)
    )
    return f'<div style="line-height:1.9;margin-bottom:4px;">{chips}</div>'


def hbar(df: pd.DataFrame, value: str, label: str, color=SEQUENTIAL_HUE,
         value_title: str | None = None, fmt: str = ",.3f") -> alt.Chart:
    base_chart = alt.Chart(df).encode(
        x=alt.X(value, title=value_title or value),
        y=alt.Y(label, sort="-x", title=None),
        tooltip=list(df.columns),
    )
    return (
        base_chart.mark_bar(color=color, cornerRadiusEnd=4, height={"band": 0.6})
        + base_chart.mark_text(align="left", dx=4, color=MUTED).encode(
            text=alt.Text(value, format=fmt)
        )
    ).properties(padding={"left": 5, "right": 52, "top": 5, "bottom": 5})


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
<div class="hdr">
  <div class="hdr-left">
    <div class="hdr-logo"><i></i><i></i><i></i></div>
    <div>
      <div class="wordmark">Smart Retail Intelligence</div>
      <div class="hdr-sub">Bangladesh FMCG distribution network · 719,817 GPS-tagged outlets</div>
    </div>
  </div>
  <div class="hdr-chips">
    <span class="hdr-chip">MITE 431 · Big Data Analytics</span>
    <span class="hdr-chip">Mendeley · CC BY 4.0</span>
    <span class="hdr-chip">R. Sikder &amp; F. Rahman</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

_m = art.get("metrics", {})
if _m:
    _q = _m.get("phase_1_preprocessing", {})
    _c = _m.get("phase_2_clustering", {})
    _r = _m.get("phase_3_regression", {})
    _cl = _m.get("phase_4_classification", {})
    _dur = sum(_m.get("durations_sec", {}).values())
    _reg_best = _r.get("models", {}).get(_r.get("best_model", ""), {})
    _clf_best = _cl.get("models", {}).get(_cl.get("best_model", ""), {})
    _pills = [
        ("outlets", f"{_q.get('n_final_shops', 0):,}"),
        ("features", _q.get("n_feature_columns", "—")),
        ("archetypes", f"k = {_c.get('best_k', '—')}"),
        ("sales R²", f"{_reg_best.get('r2', float('nan')):.2f}"),
        ("risk AUC", f"{_clf_best.get('roc_auc', float('nan')):.3f}"),
        ("full run", f"{_dur:.0f} s"),
    ]
    st.markdown(
        '<div class="readout"><span class="readout-tag">Latest run</span>'
        + "".join(f'<span class="readout-pill">{label} <b>{value}</b></span>'
                  for label, value in _pills)
        + "</div>",
        unsafe_allow_html=True,
    )

if "features" not in art:
    require(art, "features")
    st.stop()

base = build_base(tuple(sorted(art.keys())))

with st.container(border=True):
    st.markdown('<div class="eyebrow">Filters — apply to every analysis view</div>',
                unsafe_allow_html=True)
    fc1, fc2, fc3, fc4, fc5 = st.columns([2, 2, 2, 2, 1.6])
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
    only_target = fc5.toggle("Has sales target")

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

filters_active = len(f) != len(base)

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

    section_header(
        "The network", "Dataset at a glance",
        "Full dataset, independent of filters — Mendeley mgzvngzng2 v1 "
        "(Shahin, Swarna & Farid, 2025), collected Feb 2022 – Oct 2024.",
    )
    n_products_cat = len(art["products"]) if "products" in art else 42
    brands = art["products"]["brand"].nunique() if "products" in art else 5
    types_ = art["products"]["type"].nunique() if "products" in art else 9
    stat_row([
        ("Raw records", f"{quality.get('n_raw_shop_rows', len(base)):,}", "3 source files"),
        ("Outlets analysed", f"{quality.get('n_final_shops', len(base)):,}",
         f"{quality.get('n_invalid_or_missing_gps', 0):,} lacked GPS"),
        ("With sales target", f"{quality.get('n_shops_with_target', int(base['has_target'].sum())):,}",
         f"{quality.get('n_shops_with_target', 0) / max(quality.get('n_final_shops', 1), 1):.0%} of outlets"),
        ("SKU variants", n_products_cat, f"{brands} brands · {types_} types"),
        ("Targeted products", len(schema.PRODUCT_TARGET_COLUMNS), "per shop"),
    ])
    stat_row([
        ("Divisions", base["division"].nunique(), "incl. 'Unknown'"),
        ("Districts · zila", base["zila"].nunique(), None),
        ("Sub-districts · upazila", base["upazila"].nunique(), None),
        ("Model features", quality.get("n_feature_columns", "—"), "engineered in phase 1"),
        ("Mean predicted risk", f"{base['risk_score'].mean():.1%}" if "risk_score" in base else "—",
         "across scored shops"),
    ])

    if filters_active:
        st.markdown('<div style="height:0.4rem"></div>', unsafe_allow_html=True)
        section_header("Filtered view", "What the current filters select", "")
        stat_row(
            [
                ("Shops in view", f"{len(f):,}", f"{len(f) / len(base):.1%} of the network"),
                ("Divisions", f["division"].nunique(), None),
                ("Districts", f["zila"].nunique(), None),
                ("With target", f"{int(f['has_target'].sum()):,}", f"{f['has_target'].mean():.0%} coverage"),
                ("Mean predicted risk", f"{f['risk_score'].mean():.1%}" if "risk_score" in f else "—", None),
            ],
            accent=True,
        )

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
        total_rows = record["non_null"] + record["missing"]
        if record["unique"] >= 0.95 * max(record["non_null"], 1):
            st.markdown(
                f"**Identifier column** — {record['unique']:,} distinct values "
                f"across {total_rows:,} rows (effectively one per record), so a "
                "value distribution carries no information. Used as the join key."
            )
            return
        st.markdown("**Category landscape**")
        stat_row([
            ("Distinct values", f"{record['unique']:,}", None),
            ("Most common", record.get("most_common", "—"),
             f"{record.get('most_common_count', 0):,} rows"),
            ("Least common", record.get("least_common", "—"),
             f"{record.get('least_common_count', 0):,} rows"),
        ])
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
    section_header("The data", "In-depth dataset description",
                   "Every column of every source file, profiled from the actual data.")
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
            "Pick a file, then a column — meaning, central tendency, "
            "distribution, and how the pipeline treats its gaps."
        )
        st.info(
            "**What is the 'Unknown' shop type?** 85,167 records (10.23%) have "
            "no shop_type in the raw file — the survey never captured it, and "
            "the same records mostly lack style, importance and organizer. "
            "Phase 1 keeps them under an explicit **'Unknown'** label; "
            "clustering then isolates them as the data-poor 'ghost-shop' "
            "archetype of RQ1. The gap itself is a finding.",
            icon="❓",
        )

        file_tabs = st.tabs(list(profile["files"]))
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
ALGORITHM_CARDS = [
    {
        "phase": "Phase 1", "rq": "Preparation",
        "title": "Cleaning & feature engineering",
        "algorithm": "BallTree (haversine) · vectorized geometry · one-hot encoding",
        "why": "Density and centrality turn 'urban over-stock vs rural stock-out' "
               "into measurable inputs; BallTree answers all 719,817 neighbour "
               "queries in seconds (n log n, not n²).",
        "input": "shop_info_with_geo.csv (832,881 rows) + product_target_for_shop.csv (853,732 rows)",
        "output": "features.parquet — 719,817 shops × 48 features",
        "steps": [
            "Drop duplicate shop_code rows (keep first).",
            "Validate GPS against the Bangladesh bounding box (20.5–26.7°N, 88.0–92.7°E); "
            "rows with geometry = POINT EMPTY are removed.",
            "Fill categorical gaps with an explicit 'Unknown' level.",
            "Count neighbours within 500 m / 1 km / 5 km on a haversine BallTree.",
            "Compute distance to the shop centroid of each district.",
            "Join monthly targets, preferring valued rows over NaN duplicates; "
            "derive n_products_targeted and has_target.",
            "One-hot encode type, style, importance, place nature, division.",
        ],
    },
    {
        "phase": "Phase 2", "rq": "RQ1",
        "title": "Archetype clustering",
        "algorithm": "K-Means (selected) · Agglomerative + DBSCAN cross-checks",
        "why": "k is not assumed: every k from 3–9 is scored on two independent "
               "criteria, and both point to 8. DBSCAN collapsing to one cluster "
               "(98.7% noise) is reported — the network is a continuous fabric.",
        "input": "Standard-scaled 48-feature matrix (719,817 × 48)",
        "output": "cluster per shop · cluster_profiles.csv · kmeans model",
        "steps": [
            "Standardize every feature (zero mean, unit variance).",
            "Fit K-Means for k = 3…9, 10 restarts each (seed 42).",
            "Score each k: silhouette on a 10k subsample + Davies–Bouldin on all rows.",
            "Pick the best-silhouette k → 8.",
            "Cross-check with Agglomerative (Ward) and DBSCAN on a 20k subsample.",
            "Profile each cluster: median density, distance, target, dominant categories.",
        ],
    },
    {
        "phase": "Phase 3", "rq": "RQ2",
        "title": "Sales-target regression",
        "algorithm": "Linear baseline vs Random Forest · XGBoost · LightGBM + SHAP",
        "why": "The linear baseline exists to prove non-linearity (R² 0.02 vs "
               "XGBoost 0.45 on identical inputs); SHAP opens the winning "
               "black box into per-feature contributions.",
        "input": "Features of the 613,660 shops with a target; label = total_target",
        "output": "predictions.parquet · shap_importance.csv · best model",
        "steps": [
            "Split 80/20 train/test (seed 42).",
            "Fit LinearRegression (scaled), RandomForest (100 trees, depth ≤ 20), "
            "XGBoost (300 × depth 8), LightGBM (300 trees, 63 leaves).",
            "Evaluate RMSE / MAE / R² on the untouched test set.",
            "Select the lowest-RMSE model → XGBoost.",
            "Explain it with SHAP TreeExplainer on ≤ 2,000 test rows.",
            "Predict every labelled shop with the winner.",
        ],
    },
    {
        "phase": "Phase 4", "rq": "RQ3",
        "title": "Target-miss risk classification",
        "algorithm": "Class-weighted Logistic Regression · Random Forest · XGBoost",
        "why": "The dataset has assigned targets, not realized sales, so the label "
               "is an explicit peer-relative proxy. Class weights handle the 37/63 "
               "imbalance with real rows only — we measured SMOTE and it was worse.",
        "input": "Features + cluster id; label: total_target < 0.8 × peer median (cluster × type)",
        "output": "risk.parquet (at_risk + risk_score per shop) · best model",
        "steps": [
            "Build the peer-relative label per (cluster × shop type) → 36.6% base rate.",
            "Stratified 80/20 split (seed 42).",
            "Weight the minority class (scale_pos_weight ≈ 1.7 / class_weight=balanced): "
            "the loss changes, the data never does.",
            "Fit LogisticRegression (scaled), RandomForest, XGBoost.",
            "Score ROC-AUC / F1 / precision / recall on the raw test set.",
            "Pick the highest-AUC model; predict_proba → risk_score ∈ [0, 1].",
        ],
    },
    {
        "phase": "Phase 5", "rq": "RQ4",
        "title": "Per-cluster anomaly detection",
        "algorithm": "Isolation Forest + Local Outlier Factor, combined by rank",
        "why": "Scores are computed within each archetype, never globally — a rural "
               "kiosk judged against metro superstores would always look anomalous. "
               "Two detectors catch two different failure shapes.",
        "input": "Features + cluster labels for all 719,817 shops",
        "output": "anomalies.parquet · top_anomalies.parquet (71,982 ranked shops)",
        "steps": [
            "For each cluster with ≥ 20 shops, standardize its feature block.",
            "Isolation Forest (200 trees): negate score_samples, rank to percentile.",
            "LOF (20 neighbours): same negate-and-rank treatment.",
            "combined_score = mean of the two percentile ranks.",
            "Export the top decile with full context for fraud / data-error / "
            "opportunity triage.",
        ],
    },
]


def render_algorithm_card(card: dict) -> None:
    with st.container(border=True):
        st.markdown(
            f'<div class="eyebrow">{card["phase"]} · {card["rq"]}</div>'
            f'<p class="section-title" style="font-size:1.02rem;">{card["title"]}</p>'
            f'<p class="section-desc" style="margin-bottom:0.45rem;">{card["algorithm"]}</p>'
            f'<div class="stat-label" style="white-space:normal;">IN&nbsp;&nbsp;'
            f'<span style="text-transform:none;letter-spacing:0;color:#1C2A3A;">{card["input"]}</span></div>'
            f'<div class="stat-label" style="white-space:normal;margin-top:2px;">OUT&nbsp;'
            f'<span style="text-transform:none;letter-spacing:0;color:#1C2A3A;">{card["output"]}</span></div>',
            unsafe_allow_html=True,
        )
        with st.expander(f"Internal steps ({len(card['steps'])})"):
            for i, step in enumerate(card["steps"], start=1):
                st.markdown(f"{i}. {step}")
        st.caption(card["why"])


with tab_method:
    section_header(
        "Methodology", "How the analysis is produced",
        "One topic per view — pipeline flow, the algorithm behind every step, "
        "the engineered features, the distributed run, and the experiments.",
    )
    (sub_flow, sub_algo, sub_feat, sub_dist, sub_exp) = st.tabs(
        ["Pipeline flow", "Algorithms step by step", "Feature engineering",
         "Hadoop · Spark", "Experiments"]
    )

    # ---------------------------------------------------------- flow
    with sub_flow:
        st.graphviz_chart("""
digraph pipeline {
  rankdir=LR; bgcolor=transparent; compound=true;
  node [shape=box, style="rounded,filled", fillcolor="#eef4fb", color="#2a78d6",
        fontname="Helvetica", fontsize=11];
  edge [color="#8a8a8a", fontname="Helvetica", fontsize=9];
  raw [label="Raw CSVs\\n832,881 shops · 42 SKUs", fillcolor="#f5f7fa", color="#5B6B7C"];
  dash [label="Dashboard + report", fillcolor="#e8f5ee", color="#1baf7a"];

  subgraph cluster_ref {
    label="Reference lane — scikit-learn (exact proposal algorithms)";
    fontname="Helvetica"; fontsize=10; color="#c9d6e6"; style=rounded;
    p1 [label="Phase 1\\ncleaning + features"];
    p2 [label="Phase 2 (RQ1)\\nclustering"];
    p3 [label="Phase 3 (RQ2)\\nregression"];
    p4 [label="Phase 4 (RQ3)\\nrisk classification"];
    p5 [label="Phase 5 (RQ4)\\nanomaly detection"];
    p1 -> p2; p2 -> p3; p2 -> p4; p2 -> p5;
  }

  subgraph cluster_dist {
    label="Distributed lane — Hadoop + Spark (same phases, MLlib)";
    fontname="Helvetica"; fontsize=10; color="#eeddb3"; style=rounded;
    hdfs [label="HDFS storage\\nraw + division-partitioned\\nwarehouse", fillcolor="#fdf3e2", color="#eda100"];
    sp [label="Spark MLlib\\nphases 1–5, distributed", fillcolor="#fdf3e2", color="#eda100"];
    hdfs -> sp;
  }

  raw -> p1;
  raw -> hdfs [label="ingest"];
  p3 -> dash; p4 -> dash; p5 -> dash;
  sp -> dash [label="cross-validates the reference"];
}
""")
        st.caption(
            "The same five phases run in both lanes: scikit-learn is the "
            "reference (exact proposal algorithms); the Hadoop/Spark lane "
            "re-executes them distributed, storing everything in HDFS — "
            "matching row counts across lanes cross-validate the preparation."
        )
        if require(art, "metrics"):
            quality = art["metrics"].get("phase_1_preprocessing", {})
            st.markdown("**From raw file to model-ready table**")
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
                alt.Chart(funnel).mark_text(align="left", dx=4, color=MUTED)
                .encode(x="shops", y=alt.Y("stage", sort=None),
                        text=alt.Text("shops", format=","))
            )
            st.altair_chart(
                (bars + labels).properties(height=210, padding={"right": 52}),
                width="stretch",
            )
            st.caption(
                f"Stage 3 removes {quality.get('n_invalid_or_missing_gps', 0):,} rows "
                "whose coordinates were never captured (geometry = POINT EMPTY); "
                "what remains equals the paper's published count exactly."
            )

    # ---------------------------------------------------------- algorithms
    with sub_algo:
        col_a, col_b = st.columns(2)
        for slot, card in zip([col_a, col_b, col_a, col_b, col_a], ALGORITHM_CARDS):
            with slot:
                render_algorithm_card(card)

    # ---------------------------------------------------------- features
    with sub_feat:
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
            stat_row([("Model features", len(feature_cols), "engineered in phase 1")])
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

    # ---------------------------------------------------------- distributed
    with sub_dist:
        topology = cluster_topology()
        if topology:
            stat_row([
                ("Spark master", topology["spark_status"], None),
                ("Live datanodes", len(topology["datanodes"]), "HDFS, replication 2"),
                ("Live workers", len(topology["workers"]),
                 f"{sum(w['cores'] for w in topology['workers'])} cores"),
            ])
            t2, t3 = st.columns(2)
            with t2:
                st.markdown("**Datanodes**")
                st.dataframe(pd.DataFrame(topology["datanodes"]),
                             width="stretch", hide_index=True)
            with t3:
                st.markdown("**Spark workers**")
                st.dataframe(pd.DataFrame(topology["workers"]),
                             width="stretch", hide_index=True)
            st.caption("Live from the namenode and Spark master APIs · scale with "
                       "`WORKERS=N DATANODES=N make spark-up`")
        else:
            st.caption("Cluster offline — `make spark-up` to see the live topology here.")

        spark_runs = art.get("spark_runs")
        if not spark_runs:
            st.info("No distributed-run metrics yet — run `make spark-pipeline`.")
        else:
            scope = st.selectbox("Run scope", list(spark_runs.keys()))
            spark_metrics = spark_runs[scope]
            run_scope = spark_metrics.get("run_scope", {})
            st.caption(
                f"Storage `{spark_metrics.get('storage', '')}` · input: "
                f"{run_scope.get('input_source', 'raw CSVs')} · divisions: "
                f"{run_scope.get('divisions', 'all')}"
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

            with st.expander("Full metrics (JSON)"):
                st.json(spark_metrics)
            st.caption(
                "MLlib has no XGBoost/SHAP/Isolation-Forest — the distributed "
                "run uses the nearest equivalents; matching row counts across both "
                "paths cross-validate the data preparation."
            )

    # ---------------------------------------------------------- experiments
    with sub_exp:
        sensitivity = art.get("imputation_sensitivity")
        if not sensitivity:
            st.info("Run the pipeline with `--impute-knn` to populate this experiment.")
        else:
            section_header("Sensitivity", "Missing values — the k-NN imputation experiment",
                           "Gaps stay 'Unknown' by default; this measures what filling them changes.")
            ref, imp = sensitivity["reference_no_imputation"], sensitivity["with_knn_imputation"]
            details = sensitivity["imputation_details"]
            st.caption(
                f"Filled from each shop's {details['k']} nearest neighbours: "
                f"{details['imputed']['shop_style']:,} styles (≥{details['confidence_threshold']:.0%} "
                f"agreement) and {details['imputed']['division']} geo fields. shop_type "
                "and shop_importance never — measured at or below the majority-class baseline."
            )
            comparison = pd.DataFrame(
                [
                    {"metric": "clustering best k", "reference": ref["best_k"], "with imputation": imp["best_k"]},
                    {"metric": "silhouette", "reference": round(ref["silhouette"], 3), "with imputation": round(imp["silhouette"], 3)},
                    {"metric": "regression R² (XGBoost)", "reference": round(ref["regression_r2"], 3), "with imputation": round(imp["regression_r2"], 3)},
                    {"metric": "regression RMSE", "reference": round(ref["regression_rmse"], 1), "with imputation": round(imp["regression_rmse"], 1)},
                    {"metric": "risk ROC-AUC (XGBoost)", "reference": round(ref["roc_auc"], 3), "with imputation": round(imp["roc_auc"], 3)},
                    {"metric": "risk F1", "reference": round(ref["f1"], 3), "with imputation": round(imp["f1"], 3)},
                ]
            )
            st.dataframe(comparison, width="stretch", hide_index=True)
            st.markdown(
                "**The trade-off.** Imputation lifts the supervised metrics "
                "(risk AUC 0.862 → 0.886) but collapses the taxonomy from 8 "
                "archetypes to 3 — the data-poor 'ghost' segment blends into "
                "ordinary clusters, hiding the population the audit needs. The "
                "reference pipeline keeps imputation off."
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

        section_header("RQ1 · Result", f"{best_k} shop archetypes")
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

        section_header("RQ2 · Result", f"Best model: {reg.get('best_model', '?')}")
        models = pd.DataFrame(reg.get("models", {})).T.reset_index(names="model")
        st.dataframe(
            models.style.format({"rmse": "{:.3f}", "mae": "{:.3f}", "r2": "{:.3f}"}),
            width="stretch", hide_index=True,
        )
        if reg.get("cv"):
            cv = reg["cv"]
            line = (
                f"**{cv['folds']}-fold cross-validation** — best split model: "
                f"R² {cv['r2']['mean']:.3f} ± {cv['r2']['std']:.3f} · "
                f"RMSE {cv['rmse']['mean']:.1f} ± {cv['rmse']['std']:.1f}"
            )
            if reg.get("cv_tuned"):
                ct = reg["cv_tuned"]
                line += (
                    f" · tuned XGBoost: R² {ct['r2']['mean']:.3f} ± {ct['r2']['std']:.3f} · "
                    f"RMSE {ct['rmse']['mean']:.1f} ± {ct['rmse']['std']:.1f}"
                )
            st.markdown(line)
        if reg.get("tuning"):
            t = reg["tuning"]
            with st.expander(f"Hyperparameter tuning — {t['engine']}, {t['n_trials']} trials"):
                st.caption(f"{t['objective']} on {t['tuning_sample']:,} sampled shops · "
                           f"best trial RMSE {t['best_cv_rmse']}")
                st.json(t["best_params"])
        if art.get("alternatives"):
            with st.expander("Why gradient-boosted trees and not k-NN / SVR / neural nets — measured, not assumed", expanded=False):
                alt_rows = [
                    {"model": name, "train rows": f"{r['train_rows']:,}",
                     "test R²": round(r["r2"], 3), "RMSE": round(r["rmse"], 1),
                     "fit (s)": r["fit_seconds"], "predict (s)": r["predict_seconds"]}
                    for name, r in art["alternatives"].items()
                ]
                st.dataframe(pd.DataFrame(alt_rows), width="stretch", hide_index=True)
                pts = pd.DataFrame(
                    [{"model": n, "fit_seconds": max(r["fit_seconds"], 0.05),
                      "r2": r["r2"],
                      "family": "tree ensemble" if n in ("Random Forest", "XGBoost", "LightGBM")
                      else "rejected alternative"}
                     for n, r in art["alternatives"].items()]
                )
                scatter = (
                    alt.Chart(pts)
                    .mark_circle(size=90)
                    .encode(
                        x=alt.X("fit_seconds", scale=alt.Scale(type="log"),
                                title="fit time on the common 50k subsample (s, log)"),
                        y=alt.Y("r2", title="test R²"),
                        color=alt.Color("family", scale=alt.Scale(
                            domain=["tree ensemble", "rejected alternative"],
                            range=[SEQUENTIAL_HUE, "#8aa0b4"])),
                        tooltip=["model", "r2", "fit_seconds"],
                    )
                )
                labels_ = scatter.mark_text(align="left", dx=7, color=MUTED).encode(text="model")
                st.altair_chart((scatter + labels_).properties(height=260), width="stretch")
                st.caption(
                    "Every family trained on the same 50,000-shop subsample, scored on the "
                    "same 12,500 held-out shops. k-NN and the MLP sit at the linear "
                    "baseline; kernel SVR could not be trained past 10,000 rows. Only the "
                    "tree ensembles rise above the baseline — the choice is measured, not assumed."
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
                "and XGBoost are trained on a stratified 80/20 split with class "
                "weighting — the imbalance is handled in the loss function, so "
                "every training row stays a real shop."
            )

        section_header("RQ3 · Result", f"Best model: {clf.get('best_model', '?')}")
        models = pd.DataFrame(clf.get("models", {})).T.reset_index(names="model")
        st.dataframe(
            models.style.format("{:.3f}", subset=models.columns[1:].tolist()),
            width="stretch", hide_index=True,
        )
        if clf.get("cv"):
            cv = clf["cv"]
            line = (
                f"**{cv['folds']}-fold cross-validation** — best split model: "
                f"ROC-AUC {cv['roc_auc']['mean']:.3f} ± {cv['roc_auc']['std']:.3f} · "
                f"F1 {cv['f1']['mean']:.3f} ± {cv['f1']['std']:.3f}"
            )
            if clf.get("cv_tuned"):
                ct = clf["cv_tuned"]
                line += (
                    f" · tuned XGBoost: ROC-AUC {ct['roc_auc']['mean']:.3f} ± {ct['roc_auc']['std']:.3f} · "
                    f"F1 {ct['f1']['mean']:.3f} ± {ct['f1']['std']:.3f}"
                )
            st.markdown(line)
        if clf.get("tuning"):
            t = clf["tuning"]
            with st.expander(f"Hyperparameter tuning — {t['engine']}, {t['n_trials']} trials"):
                st.caption(f"{t['objective']} on {t['tuning_sample']:,} sampled shops · "
                           f"best trial ROC-AUC {t['best_cv_roc_auc']}")
                st.json(t["best_params"])
        if art.get("imbalance"):
            with st.expander("Why this balancing strategy — all four measured on the same split and model"):
                imb_rows = [
                    {"strategy": name, "ROC-AUC": round(r["roc_auc"], 3),
                     "F1": round(r["f1"], 3), "recall": round(r["recall"], 3),
                     "fit (s)": r["fit_seconds"]}
                    for name, r in art["imbalance"].items()
                ]
                st.dataframe(pd.DataFrame(imb_rows), width="stretch", hide_index=True)
                st.caption(
                    "Same XGBoost, same stratified split. Class weights match SMOTE's AUC while "
                    "beating it on recall and F1, run 5.6× faster, and use only real shops — "
                    "no synthetic rows anywhere. That is why both lanes (scikit-learn and "
                    "Spark MLlib) weight the loss instead of resampling; SMOTE was measured "
                    "and rejected."
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

        section_header("RQ4 · Result", "Ranked anomalies")
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
        section_header("Report", "Dataset to conclusions, end to end")
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
