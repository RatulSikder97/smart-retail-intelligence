"""Country map of the flagged anomalies plus per-division anomaly share; writes f19_anomaly_map.png."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

features = pd.read_parquet(ROOT / "data/processed/features.parquet",
                           columns=["shop_code", "latitude", "longitude", "division"])
top = pd.read_parquet(ROOT / "data/processed/top_anomalies.parquet")

share = (top.groupby("division").size() / features.groupby("division").size()).dropna()
share = share.sort_values(ascending=False)
(ROOT / "report/anomaly_share_by_division.json").write_text(
    json.dumps({k: round(float(v), 4) for k, v in share.items()}, indent=2))
print(share.round(4).to_string())

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e2e7ec",
                     "grid.linewidth": 0.5, "axes.axisbelow": True, "savefig.dpi": 300})
fig, ax = plt.subplots(figsize=(6.2, 4.8))
bg = features.sample(60_000, random_state=SEED)
ax.scatter(bg["longitude"], bg["latitude"], s=0.9, color="#b7c2cd", linewidths=0,
           rasterized=True)  # background context; explained in the caption
an = top.sample(20_000, random_state=SEED).sort_values("combined_score")
sc = ax.scatter(an["longitude"], an["latitude"], s=2.2, c=an["combined_score"],
                cmap="YlOrRd", vmin=0.79, vmax=1.0, linewidths=0, rasterized=True)
cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.03)
cb.set_label("combined anomaly score", fontsize=8.5)
cb.ax.tick_params(labelsize=8)
ax.set_xlabel("longitude")
ax.set_ylabel("latitude")
ax.set_aspect(1.08)
fig.savefig(ROOT / "report/figures/f19_anomaly_map.png",
            bbox_inches="tight", pad_inches=0.02)
print("wrote f19_anomaly_map.png")
