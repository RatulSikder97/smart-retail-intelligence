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
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "axes.grid": True,
    "grid.color": "#e2e7ec",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})
COL_W = 4.3   # figures render 1:1 at the report display width
WIDE_W = 6.2  # full text-block width for two-panel figures and diagrams
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
fig, ax = plt.subplots(figsize=(COL_W, 2.6))
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
fig, ax = plt.subplots(figsize=(COL_W, 2.4))
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
fig, axes = plt.subplots(2, 1, figsize=(COL_W, 3.4), sharex=True)
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
fig, ax = plt.subplots(figsize=(COL_W, 2.4))
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
fig, ax = plt.subplots(figsize=(COL_W, 2.75))
ax.barh(top["feature"], top["mean_abs_shap"], color=BLUE, height=0.62, zorder=3)
ax.set_xlabel("mean |SHAP value|")
ax.grid(axis="y", visible=False)
save(fig, "f5_shap.png")

# F6 — observed at-risk rate by division
merged = risk.merge(features[["shop_code", "division"]], on="shop_code")
rate = (merged.groupby("division")["at_risk"].mean()
        .drop("Unknown", errors="ignore").sort_values())
fig, ax = plt.subplots(figsize=(COL_W, 2.5))
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
fig, ax = plt.subplots(figsize=(COL_W, 2.1))
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
fig, ax = plt.subplots(figsize=(COL_W, 2.9))
ax.barh(tc.index, tc.values, color=BLUE, height=0.62, zorder=3)
for i, v in enumerate(tc.values):
    ax.text(v + tc.max() * 0.015, i, f"{v:,}", va="center", fontsize=7)
ax.set_xlabel("shops"); ax.set_xlim(0, tc.max() * 1.22)
ax.xaxis.set_major_formatter(compact); ax.grid(axis="y", visible=False)
save(fig, "f8_shop_types.png")

# F9 — engineered feature distributions (2x2 panel)
fig, axes = plt.subplots(2, 2, figsize=(WIDE_W, 3.2))
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
fig, ax = plt.subplots(figsize=(COL_W, 2.75))
ax.barh(miss.index, miss.values, color=RED, height=0.6, zorder=3)
for i, v in enumerate(miss.values):
    ax.text(v + 1.2, i, f"{v:.1f}%", va="center", fontsize=7)
ax.set_xlabel("missing (%)"); ax.set_xlim(0, 112)
ax.grid(axis="y", visible=False)
save(fig, "f10_missingness.png")

# F11 — spatial map of archetypes (sampled scatter)
merged = features_full.merge(clusters, on="shop_code")
sample = merged.sample(60_000, random_state=42)
fig, ax = plt.subplots(figsize=(COL_W, 3.9))
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
fig, ax = plt.subplots(figsize=(COL_W, 3.6))
ax.scatter(preds["actual"], preds["predicted"], s=1.6, alpha=0.25, color=BLUE,
           rasterized=True, linewidths=0)
ax.plot([0, cap], [0, cap], color=RED, lw=1.0, ls="--", label="perfect prediction")
ax.set_xlim(0, cap); ax.set_ylim(0, cap)
ax.set_xlabel("assigned monthly target"); ax.set_ylabel("model prediction")
ax.legend(frameon=False, loc="upper left")
save(fig, "f13_actual_vs_predicted.png")

# F14 — anomaly combined-score distribution
anom = pd.read_parquet(ROOT / "data/processed/anomalies.parquet", columns=["combined_score"])
scores = anom["combined_score"].dropna()
decile_cut = scores.quantile(0.90)  # empirical, not 0.90: the mean of two ranks concentrates
fig, ax = plt.subplots(figsize=(COL_W, 2.25))
ax.hist(scores, bins=50, color=BLUE, zorder=3)
ax.axvline(decile_cut, color=RED, lw=0.9,
           label=f"top-decile threshold ({decile_cut:.2f})")
ax.set_xlabel("combined anomaly score")
ax.set_ylabel("shops"); ax.yaxis.set_major_formatter(compact)
ax.legend(frameon=False)
save(fig, "f14_anomaly_scores.png")
print("extra figures written")

# F15 — pipeline architecture (two lanes), plain paper style
import matplotlib.patches as mpatches

def _diagram_axes(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.grid(False)
    return fig, ax

def _pbox(ax, x, y, w, h, title, body=None, title_fs=8, body_fs=7, lw=0.9):
    ax.add_patch(mpatches.Rectangle((x, y), w, h, fc="white", ec="black", lw=lw))
    if body:
        # centred anchors keep title and body apart at any box size
        ax.text(x + w / 2, y + h * 0.76, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color="black")
        ax.text(x + w / 2, y + h * 0.33, body, ha="center", va="center",
                fontsize=body_fs, color="black", linespacing=1.55)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color="black")

def _parrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.9,
                                shrinkA=0, shrinkB=0))

fig, ax = _diagram_axes(6.5, 2.9)

# dashed lane containers with italic labels, as in a journal figure
ax.add_patch(mpatches.Rectangle((0.19, 0.53), 0.60, 0.45, fc="none", ec="black",
                                lw=0.7, ls=(0, (4, 3))))
ax.text(0.20, 0.965, "Reference lane (scikit-learn — exact proposal algorithms)",
        ha="left", va="top", fontsize=7.5, style="italic", color="black")
ax.add_patch(mpatches.Rectangle((0.19, 0.03), 0.60, 0.45, fc="none", ec="black",
                                lw=0.7, ls=(0, (4, 3))))
ax.text(0.20, 0.465, "Distributed lane (Hadoop HDFS + Apache Spark — same phases)",
        ha="left", va="top", fontsize=7.5, style="italic", color="black")

_pbox(ax, 0.010, 0.365, 0.140, 0.28, "Raw dataset", "3 CSVs · 206 MB\n832,881 rows")
_pbox(ax, 0.845, 0.365, 0.145, 0.28, "Deliverables", "dashboard · report\naudit lists")

ref_y, ref_h, ref_w = 0.565, 0.30, 0.132
ref_x = [0.205, 0.352, 0.499, 0.646]
_pbox(ax, ref_x[0], ref_y, ref_w, ref_h, "Phase 1", "cleaning\n48 features")
_pbox(ax, ref_x[1], ref_y, ref_w, ref_h, "Phase 2", "K-Means\nk = 8")
_pbox(ax, ref_x[2], ref_y, ref_w, ref_h, "Phases 3–4", "XGBoost\nweights · SHAP")
_pbox(ax, ref_x[3], ref_y, ref_w, ref_h, "Phase 5", "IForest\n+ LOF")
for a, b in zip(ref_x[:-1], ref_x[1:]):
    _parrow(ax, a + ref_w, ref_y + ref_h / 2, b, ref_y + ref_h / 2)

d_y, d_h, d_w = 0.075, 0.30, 0.182
d_x = [0.205, 0.402, 0.599]
_pbox(ax, d_x[0], d_y, d_w, d_h, "HDFS warehouse", "9 datanodes · repl. 2\ndivision partitions")
_pbox(ax, d_x[1], d_y, d_w, d_h, "Spark SQL", "distributed cleaning\ngrid densities")
_pbox(ax, d_x[2], d_y, d_w, d_h, "Spark MLlib", "phases 2–5\nregional runs")
for a, b in zip(d_x[:-1], d_x[1:]):
    _parrow(ax, a + d_w, d_y + d_h / 2, b, d_y + d_h / 2)

_parrow(ax, 0.150, 0.56, 0.205, 0.71)
_parrow(ax, 0.150, 0.45, 0.205, 0.22)
_parrow(ax, 0.778, 0.71, 0.845, 0.56)
_parrow(ax, 0.781, 0.22, 0.845, 0.45)
save(fig, "f15_architecture.png")

# F0 — overall methodology: six numbered steps, plain paper style
steps = [
    ("1. Dataset", "Mendeley data\n832,881 rows"),
    ("2. Ingest", "HDFS, 9 nodes\nby division"),
    ("3. Engineer", "GPS checks\n48 features"),
    ("4. Model", "RQ1–RQ4\nfour methods"),
    ("5. Evaluate", "Optuna, 5-fold\n47 tests"),
    ("6. Deliver", "dashboard\naudit lists"),
]
fig, ax = _diagram_axes(6.5, 1.15)
bw, gap = 0.148, 0.0208
x = 0.004
for i, (title, body) in enumerate(steps):
    _pbox(ax, x, 0.10, bw, 0.80, title, body, title_fs=6.8, body_fs=6.4)
    if i < len(steps) - 1:
        _parrow(ax, x + bw, 0.50, x + bw + gap, 0.50)
    x += bw + gap
save(fig, "f_overview.png")
print("architecture figure written")

# ------------------------------------------------ professional EDA upgrades
# F2 (regenerated): two-panel target distribution — linear (clipped) + log scale
t = features.loc[features.has_target, "total_target"]
t_pos = t[t > 0]
fig, axes = plt.subplots(1, 2, figsize=(WIDE_W, 1.9))
cap = t.quantile(0.99)
axes[0].hist(t[t <= cap], bins=60, color=BLUE, zorder=3)
axes[0].axvline(t.median(), color=RED, lw=0.9, label=f"median = {t.median():.1f}")
axes[0].axvline(t.mean(), color="#444444", lw=0.9, ls="--", label=f"mean = {t.mean():.1f}")
axes[0].set_xlabel("monthly total target (clipped at p99)")
axes[0].set_ylabel("shops"); axes[0].yaxis.set_major_formatter(compact)
axes[0].legend(frameon=False); axes[0].set_title("(a) linear scale", fontsize=9)
import numpy as np
bins = np.logspace(np.log10(max(t_pos.min(), 0.01)), np.log10(t_pos.max()), 60)
axes[1].hist(t_pos, bins=bins, color=BLUE, zorder=3)
axes[1].set_xscale("log")
axes[1].set_xlabel("monthly total target (log scale, full range)")
axes[1].set_ylabel("shops"); axes[1].yaxis.set_major_formatter(compact)
axes[1].set_title("(b) log scale — the wholesale tail", fontsize=9)
fig.tight_layout()
save(fig, "f2_target_distribution.png")

# F9 (regenerated): 2x2 with panel letters
fig, axes = plt.subplots(2, 2, figsize=(WIDE_W, 3.3))
panels = [
    ("density_1km", "(a) shops within 1 km", axes[0][0]),
    ("dist_to_district_centre_km", "(b) km to district centre", axes[0][1]),
    ("n_products_targeted", "(c) products targeted (of 17)", axes[1][0]),
]
for col, label, ax in panels:
    v = features_full[col].dropna()
    ax.hist(v[v <= v.quantile(0.99)], bins=40, color=BLUE, zorder=3)
    ax.set_title(label, fontsize=9, loc="left")
    ax.set_ylabel("shops"); ax.yaxis.set_major_formatter(compact)
ax = axes[1][1]
ax.hist(t[t <= cap], bins=40, color=BLUE, zorder=3)
ax.set_title("(d) monthly total target", fontsize=9, loc="left")
ax.set_ylabel("shops"); ax.yaxis.set_major_formatter(compact)
fig.tight_layout()
save(fig, "f9_feature_distributions.png")

# F16 — target by division: median with IQR
j = features[features.has_target].dropna(subset=["total_target"])
grp = j.groupby("division")["total_target"]
stats = pd.DataFrame({"q1": grp.quantile(0.25), "med": grp.quantile(0.5),
                      "q3": grp.quantile(0.75)}).drop("Unknown", errors="ignore")
stats = stats.sort_values("med")
fig, ax = plt.subplots(figsize=(COL_W, 2.9))
y = range(len(stats))
ax.hlines(y, stats["q1"], stats["q3"], color="#b9cde4", lw=5, zorder=2)
ax.plot(stats["med"], y, "o", color=BLUE, ms=5, zorder=3)
for i, (med, q3) in enumerate(zip(stats["med"], stats["q3"])):
    ax.text(q3 + 0.6, i, f"{med:.1f}", va="center", fontsize=7, color="#41556E")
ax.set_yticks(list(y)); ax.set_yticklabels(stats.index)
ax.set_xlabel("monthly total target — median (dot) and IQR (band)")
ax.grid(axis="y", visible=False)
save(fig, "f16_target_by_division.png")
print("EDA upgrades written")
