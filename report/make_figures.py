"""Generate publication-quality figures for the IEEE-format report.

Sized for a single IEEE column (3.45 in), 300 dpi, serif, formatted ticks.
Run after a full pipeline execution:  .venv/bin/python report/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.grid": True,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.4,
    "axes.axisbelow": True,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})
COL_W = 3.45  # IEEE single-column width in inches
BLUE, RED = "#2a78d6", "#c0392b"
CLUSTER_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
                  "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
thousands = FuncFormatter(lambda x, _: f"{x:,.0f}")
compact = FuncFormatter(lambda x, _: f"{x/1000:,.0f}k" if x >= 1000 else f"{x:,.0f}")

metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
features = pd.read_parquet(ROOT / "data/processed/features.parquet",
                           columns=["shop_code", "division", "total_target",
                                    "has_target", "density_1km"])
clusters = pd.read_parquet(ROOT / "data/processed/clusters.parquet")
risk = pd.read_parquet(ROOT / "data/processed/risk.parquet")
shap_imp = pd.read_csv(ROOT / "reports/shap_importance.csv")


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


# F1 — shops per division
counts = features["division"].value_counts().drop("Unknown", errors="ignore").sort_values()
fig, ax = plt.subplots(figsize=(COL_W, 2.1))
ax.barh(counts.index, counts.values, color=BLUE, height=0.62, zorder=3)
for i, v in enumerate(counts.values):
    ax.text(v + counts.max() * 0.015, i, f"{v:,}", va="center", fontsize=7)
ax.set_xlabel("shops")
ax.set_xlim(0, counts.max() * 1.22)
ax.xaxis.set_major_formatter(compact)
ax.grid(axis="y", visible=False)
save(fig, "f1_division_counts.png")

# F2 — monthly target distribution (clipped at p99)
t = features.loc[features.has_target, "total_target"]
cap = t.quantile(0.99)
fig, ax = plt.subplots(figsize=(COL_W, 1.9))
ax.hist(t[t <= cap], bins=60, color=BLUE, zorder=3)
ax.axvline(t.median(), color=RED, lw=0.9,
           label=f"median = {t.median():.1f}")
ax.axvline(t.mean(), color="#444444", lw=0.9, ls="--",
           label=f"mean = {t.mean():.1f}")
ax.set_xlabel("monthly total target (clipped at 99th percentile)")
ax.set_ylabel("shops")
ax.yaxis.set_major_formatter(compact)
ax.legend(frameon=False)
save(fig, "f2_target_distribution.png")

# F3 — model selection for k (stacked panels, single column)
km = metrics["phase_2_clustering"]["kmeans"]
ks = sorted(int(k) for k in km)
sil = [km[str(k)]["silhouette"] for k in ks]
db = [km[str(k)]["davies_bouldin"] for k in ks]
best_k = metrics["phase_2_clustering"]["best_k"]
fig, axes = plt.subplots(2, 1, figsize=(COL_W, 2.7), sharex=True)
axes[0].plot(ks, sil, "o-", color=BLUE, lw=1, ms=3.5, zorder=3)
axes[0].set_ylabel("silhouette\n(higher better)")
axes[1].plot(ks, db, "o-", color=BLUE, lw=1, ms=3.5, zorder=3)
axes[1].set_ylabel("Davies–Bouldin\n(lower better)")
axes[1].set_xlabel("number of clusters k")
axes[1].set_xticks(ks)
for ax in axes:
    ax.axvline(best_k, color=RED, lw=0.8, ls=":", zorder=2)
fig.align_ylabels(axes)
fig.subplots_adjust(hspace=0.15)
save(fig, "f3_k_selection.png")

# F4 — cluster sizes
sizes = clusters["cluster"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(COL_W, 1.9))
ax.bar([f"C{i}" for i in sizes.index], sizes.values,
       color=[CLUSTER_COLORS[i % 8] for i in sizes.index], width=0.65, zorder=3)
for i, v in enumerate(sizes.values):
    ax.text(i, v + sizes.max() * 0.02, f"{v/1000:,.0f}k", ha="center", fontsize=7)
ax.set_ylabel("shops")
ax.set_ylim(0, sizes.max() * 1.16)
ax.yaxis.set_major_formatter(compact)
ax.grid(axis="x", visible=False)
save(fig, "f4_cluster_sizes.png")

# F5 — SHAP top 10
top = shap_imp.head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(COL_W, 2.2))
ax.barh(top["feature"], top["mean_abs_shap"], color=BLUE, height=0.62, zorder=3)
ax.set_xlabel("mean |SHAP value|")
ax.grid(axis="y", visible=False)
save(fig, "f5_shap.png")

# F6 — observed at-risk rate by division
merged = risk.merge(features[["shop_code", "division"]], on="shop_code")
rate = (merged.groupby("division")["at_risk"].mean()
        .drop("Unknown", errors="ignore").sort_values())
fig, ax = plt.subplots(figsize=(COL_W, 2.0))
ax.barh(rate.index, rate.values * 100, color=RED, height=0.62, zorder=3)
for i, v in enumerate(rate.values):
    ax.text(v * 100 + 0.6, i, f"{v:.1%}", va="center", fontsize=7)
ax.set_xlabel("shops labelled at-risk (%)")
ax.set_xlim(0, rate.max() * 100 * 1.18)
ax.grid(axis="y", visible=False)
save(fig, "f6_risk_by_division.png")

# F7 — preprocessing funnel
q = metrics["phase_1_preprocessing"]
stages = ["Raw records", "Unique codes", "Valid GPS", "With target"]
vals = [q["n_raw_shop_rows"], q["n_unique_shops"],
        q["n_final_shops"], q["n_shops_with_target"]]
fig, ax = plt.subplots(figsize=(COL_W, 1.7))
ax.barh(stages[::-1], vals[::-1], color=BLUE, height=0.6, zorder=3)
for i, v in enumerate(vals[::-1]):
    ax.text(v + max(vals) * 0.015, i, f"{v:,}", va="center", fontsize=7)
ax.set_xlabel("shops")
ax.set_xlim(0, max(vals) * 1.24)
ax.xaxis.set_major_formatter(compact)
ax.grid(axis="y", visible=False)
save(fig, "f7_funnel.png")

print("all figures written to", FIG)

# ---------------------------------------------------------------- extras
features_full = pd.read_parquet(ROOT / "data/processed/features.parquet",
                                columns=["shop_code", "shop_type", "latitude", "longitude",
                                         "density_1km", "dist_to_district_centre_km",
                                         "n_products_targeted"])

# F8 — shops per business type
tc = features_full["shop_type"].value_counts().head(12).sort_values()
fig, ax = plt.subplots(figsize=(COL_W, 2.3))
ax.barh(tc.index, tc.values, color=BLUE, height=0.62, zorder=3)
for i, v in enumerate(tc.values):
    ax.text(v + tc.max() * 0.015, i, f"{v:,}", va="center", fontsize=7)
ax.set_xlabel("shops"); ax.set_xlim(0, tc.max() * 1.22)
ax.xaxis.set_major_formatter(compact); ax.grid(axis="y", visible=False)
save(fig, "f8_shop_types.png")

# F9 — engineered feature distributions (2x2 panel)
fig, axes = plt.subplots(2, 2, figsize=(COL_W * 1.9, 3.4))
panels = [
    ("density_1km", "shops within 1 km", axes[0][0]),
    ("dist_to_district_centre_km", "km to district centre", axes[0][1]),
    ("n_products_targeted", "products targeted (of 17)", axes[1][0]),
]
for col, label, ax in panels:
    v = features_full[col].dropna()
    ax.hist(v[v <= v.quantile(0.99)], bins=40, color=BLUE, zorder=3)
    ax.set_xlabel(label); ax.set_ylabel("shops")
    ax.yaxis.set_major_formatter(compact)
t = features.loc[features.has_target, "total_target"]
ax = axes[1][1]
ax.hist(t[t <= t.quantile(0.99)], bins=40, color=BLUE, zorder=3)
ax.set_xlabel("monthly total target"); ax.set_ylabel("shops")
ax.yaxis.set_major_formatter(compact)
fig.tight_layout()
save(fig, "f9_feature_distributions.png")

# F10 — missingness per raw column
prof = json.loads((ROOT / "reports/dataset_profile.json").read_text())
cols = prof["files"]["shop_info_with_geo.csv"]["columns"]
miss = pd.Series({c["column"]: c["missing_pct"] for c in cols if c["missing_pct"] > 0}).sort_values()
fig, ax = plt.subplots(figsize=(COL_W, 2.2))
ax.barh(miss.index, miss.values, color=RED, height=0.6, zorder=3)
for i, v in enumerate(miss.values):
    ax.text(v + 1.2, i, f"{v:.1f}%", va="center", fontsize=7)
ax.set_xlabel("missing (%)"); ax.set_xlim(0, 112)
ax.grid(axis="y", visible=False)
save(fig, "f10_missingness.png")

# F11 — spatial map of archetypes (sampled scatter)
merged = features_full.merge(clusters, on="shop_code")
sample = merged.sample(60_000, random_state=42)
fig, ax = plt.subplots(figsize=(COL_W * 1.35, 4.1))
for c in sorted(sample["cluster"].unique()):
    s = sample[sample["cluster"] == c]
    ax.scatter(s["longitude"], s["latitude"], s=0.4, color=CLUSTER_COLORS[int(c) % 8],
               label=f"C{int(c)}", rasterized=True, linewidths=0)
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_aspect(1.09)  # ~cos(23.7 deg) correction
leg = ax.legend(frameon=False, markerscale=14, ncol=4, loc="lower right",
                columnspacing=0.8, handletextpad=0.1)
save(fig, "f11_cluster_map.png")

# F13 — actual vs predicted (best regression model)
preds = pd.read_parquet(ROOT / "data/processed/predictions.parquet").sample(30_000, random_state=42)
cap = preds[["actual", "predicted"]].quantile(0.99).max()
fig, ax = plt.subplots(figsize=(COL_W, 2.9))
ax.scatter(preds["actual"], preds["predicted"], s=0.8, alpha=0.25, color=BLUE,
           rasterized=True, linewidths=0)
ax.plot([0, cap], [0, cap], color=RED, lw=0.8, ls="--", label="perfect prediction")
ax.set_xlim(0, cap); ax.set_ylim(0, cap)
ax.set_xlabel("assigned monthly target"); ax.set_ylabel("model prediction")
ax.legend(frameon=False)
save(fig, "f13_actual_vs_predicted.png")

# F14 — anomaly combined-score distribution
anom = pd.read_parquet(ROOT / "data/processed/anomalies.parquet", columns=["combined_score"])
fig, ax = plt.subplots(figsize=(COL_W, 1.8))
ax.hist(anom["combined_score"].dropna(), bins=50, color=BLUE, zorder=3)
ax.axvline(0.90, color=RED, lw=0.9, label="top-decile threshold")
ax.set_xlabel("combined anomaly score (mean of per-cluster percentile ranks)")
ax.set_ylabel("shops"); ax.yaxis.set_major_formatter(compact)
ax.legend(frameon=False)
save(fig, "f14_anomaly_scores.png")
print("extra figures written")

# F15 — pipeline architecture (two lanes)
import matplotlib.patches as mpatches
fig, ax = plt.subplots(figsize=(COL_W * 1.9, 2.6))
ax.axis("off")

def box(x, y, w, h, text, fc="#eef4fb", ec="#2a78d6"):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                         fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.6)

def arrow(x1, y1, x2, y2, label=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#777", lw=0.7))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.02, label,
                ha="center", fontsize=5.8, color="#555")

# lanes
ax.add_patch(mpatches.FancyBboxPatch((0.16, 0.52), 0.60, 0.40, boxstyle="round,pad=0.008",
                                     fc="none", ec="#c9d6e6", lw=0.7, ls="-"))
ax.text(0.46, 0.945, "Reference lane — scikit-learn (exact proposal algorithms)",
        ha="center", fontsize=6.2, color="#41556e")
ax.add_patch(mpatches.FancyBboxPatch((0.16, 0.06), 0.60, 0.36, boxstyle="round,pad=0.008",
                                     fc="none", ec="#eeddb3", lw=0.7))
ax.text(0.46, 0.445, "Distributed lane — Hadoop HDFS + Apache Spark (same phases)",
        ha="center", fontsize=6.2, color="#8a6d1d")

box(0.01, 0.38, 0.13, 0.18, "Raw CSVs\n832,881 shops\n42 SKUs", fc="#f5f7fa", ec="#5B6B7C")
# reference lane
box(0.18, 0.66, 0.16, 0.16, "Phase 1\ncleaning +\nBallTree features")
box(0.37, 0.66, 0.14, 0.16, "Phase 2\nK-Means k=8")
box(0.54, 0.74, 0.20, 0.10, "Phase 3  regression + SHAP")
box(0.54, 0.62, 0.20, 0.10, "Phase 4  risk + SMOTE")
box(0.54, 0.545, 0.20, 0.06, "Phase 5  IForest + LOF")
# distributed lane
box(0.18, 0.16, 0.17, 0.18, "HDFS storage\n9 datanodes, repl. 2\ndivision-partitioned\nwarehouse", fc="#fdf3e2", ec="#eda100")
box(0.40, 0.16, 0.17, 0.18, "Spark SQL\ncleaning + grid\ndensities", fc="#fdf3e2", ec="#eda100")
box(0.60, 0.16, 0.14, 0.18, "Spark MLlib\nphases 2–5", fc="#fdf3e2", ec="#eda100")
box(0.80, 0.38, 0.17, 0.18, "Dashboard\n+ report", fc="#e8f5ee", ec="#1baf7a")

arrow(0.14, 0.50, 0.18, 0.72)
arrow(0.14, 0.44, 0.18, 0.28, "ingest")
arrow(0.34, 0.74, 0.37, 0.74)
arrow(0.51, 0.74, 0.54, 0.78)
arrow(0.51, 0.74, 0.54, 0.66)
arrow(0.51, 0.72, 0.54, 0.575)
arrow(0.74, 0.70, 0.82, 0.56)
arrow(0.35, 0.25, 0.40, 0.25)
arrow(0.57, 0.25, 0.60, 0.25)
arrow(0.74, 0.25, 0.82, 0.40, "cross-validates")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
save(fig, "f15_architecture.png")
print("architecture figure written")
