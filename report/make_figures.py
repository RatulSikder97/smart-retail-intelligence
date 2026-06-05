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
