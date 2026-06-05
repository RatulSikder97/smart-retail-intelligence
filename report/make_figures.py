"""Generate the report figures from the pipeline artifacts.

Run after a full pipeline execution:  .venv/bin/python report/make_figures.py
Writes PNGs (150 dpi, serif, print-friendly) into report/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "figure.dpi": 150,
})
BLUE, RED = "#2a78d6", "#e34948"
CLUSTER_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
                  "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]

metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
features = pd.read_parquet(ROOT / "data/processed/features.parquet",
                           columns=["shop_code", "division", "total_target",
                                    "has_target", "density_1km"])
clusters = pd.read_parquet(ROOT / "data/processed/clusters.parquet")
risk = pd.read_parquet(ROOT / "data/processed/risk.parquet")
shap_imp = pd.read_csv(ROOT / "reports/shap_importance.csv")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# F1 — shops per division
counts = features["division"].value_counts().sort_values()
fig, ax = plt.subplots(figsize=(6.2, 3.2))
ax.barh(counts.index, counts.values, color=BLUE)
for i, v in enumerate(counts.values):
    ax.text(v + 3000, i, f"{v:,}", va="center", fontsize=8.5)
ax.set_xlabel("shops")
ax.set_xlim(0, counts.max() * 1.18)
save(fig, "f1_division_counts.png")

# F2 — monthly target distribution (clipped at p99)
t = features.loc[features.has_target, "total_target"]
cap = t.quantile(0.99)
fig, ax = plt.subplots(figsize=(6.2, 3.0))
ax.hist(t[t <= cap], bins=60, color=BLUE, alpha=0.9)
ax.axvline(t.median(), color=RED, lw=1.2, label=f"median = {t.median():.1f}")
ax.axvline(t.mean(), color="#555", lw=1.2, ls="--", label=f"mean = {t.mean():.1f}")
ax.set_xlabel("monthly total target (clipped at the 99th percentile)")
ax.set_ylabel("shops")
ax.legend(frameon=False, fontsize=9)
save(fig, "f2_target_distribution.png")

# F3 — model selection for k
km = metrics["phase_2_clustering"]["kmeans"]
ks = sorted(int(k) for k in km)
sil = [km[str(k)]["silhouette"] for k in ks]
db = [km[str(k)]["davies_bouldin"] for k in ks]
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
axes[0].plot(ks, sil, "o-", color=BLUE)
axes[0].set_xlabel("k"); axes[0].set_ylabel("silhouette")
axes[0].set_title("higher is better", fontsize=9)
axes[1].plot(ks, db, "o-", color=BLUE)
axes[1].set_xlabel("k"); axes[1].set_ylabel("Davies–Bouldin")
axes[1].set_title("lower is better", fontsize=9)
best_k = metrics["phase_2_clustering"]["best_k"]
for ax in axes:
    ax.axvline(best_k, color=RED, lw=1, ls=":")
save(fig, "f3_k_selection.png")

# F4 — cluster sizes
sizes = clusters["cluster"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(6.2, 3.0))
ax.bar([f"C{i}" for i in sizes.index],
       sizes.values, color=[CLUSTER_COLORS[i % 8] for i in sizes.index])
for i, v in enumerate(sizes.values):
    ax.text(i, v + 3000, f"{v:,}", ha="center", fontsize=8)
ax.set_ylabel("shops"); ax.set_ylim(0, sizes.max() * 1.15)
save(fig, "f4_cluster_sizes.png")

# F5 — SHAP top 12
top = shap_imp.head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.barh(top["feature"], top["mean_abs_shap"], color=BLUE)
ax.set_xlabel("mean |SHAP value|")
save(fig, "f5_shap.png")

# F6 — observed at-risk rate by division
merged = risk.merge(features[["shop_code", "division"]], on="shop_code")
rate = merged.groupby("division")["at_risk"].mean().drop("Unknown", errors="ignore").sort_values()
fig, ax = plt.subplots(figsize=(6.2, 3.0))
ax.barh(rate.index, rate.values, color=RED, alpha=0.85)
for i, v in enumerate(rate.values):
    ax.text(v + 0.004, i, f"{v:.1%}", va="center", fontsize=8.5)
ax.set_xlabel("share of shops labelled at-risk")
ax.set_xlim(0, rate.max() * 1.2)
save(fig, "f6_risk_by_division.png")

# F7 — preprocessing funnel
q = metrics["phase_1_preprocessing"]
stages = ["Raw records", "Unique codes", "Valid GPS", "With target"]
vals = [q["n_raw_shop_rows"], q["n_unique_shops"], q["n_final_shops"], q["n_shops_with_target"]]
fig, ax = plt.subplots(figsize=(6.2, 2.6))
ax.barh(stages[::-1], vals[::-1], color=BLUE)
for i, v in enumerate(vals[::-1]):
    ax.text(v + 8000, i, f"{v:,}", va="center", fontsize=8.5)
ax.set_xlabel("shops"); ax.set_xlim(0, max(vals) * 1.18)
save(fig, "f7_funnel.png")

print("all figures written to", FIG)
