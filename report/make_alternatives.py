"""Evidence for the model-choice discussion: train the rejected alternatives.

Every candidate is fitted on the SAME 50,000-shop training subsample and
scored on the same 12,500-shop held-out subsample, recording test R2, RMSE,
fit time and predict time. Kernel SVR cannot finish even at this size, so it
is additionally fitted at 10,000 rows to document its scaling. Results land
in report/alternatives.json and the time-vs-accuracy figure.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from smart_retail.features.engineering import TARGET_COLUMN, feature_matrix

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

features = pd.read_parquet(ROOT / "data/processed/features.parquet")
labelled = features[features["has_target"]]
X = feature_matrix(labelled)
y = labelled[TARGET_COLUMN].astype(float)

X_pool, _, y_pool, _ = train_test_split(X, y, train_size=62_500, random_state=SEED)
X_tr, X_te, y_tr, y_te = train_test_split(X_pool, y_pool, train_size=50_000,
                                          random_state=SEED)
print(f"common protocol: train {len(X_tr):,}, test {len(X_te):,}")

candidates = {
    "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
    "k-NN (k=15)": make_pipeline(StandardScaler(),
                                 KNeighborsRegressor(n_neighbors=15, n_jobs=-1)),
    "MLP (2x128)": make_pipeline(StandardScaler(),
                                 MLPRegressor(hidden_layer_sizes=(128, 128),
                                              early_stopping=True, max_iter=200,
                                              random_state=SEED)),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=20,
                                           min_samples_leaf=5, n_jobs=-1,
                                           random_state=SEED),
    "XGBoost": XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.1,
                            n_jobs=-1, random_state=SEED, verbosity=0),
    "LightGBM": LGBMRegressor(n_estimators=300, num_leaves=63, learning_rate=0.1,
                              n_jobs=-1, random_state=SEED, verbose=-1),
}

results = {}
for name, model in candidates.items():
    t0 = time.time()
    model.fit(X_tr, y_tr)
    fit_s = time.time() - t0
    t0 = time.time()
    pred = model.predict(X_te)
    pred_s = time.time() - t0
    results[name] = {
        "train_rows": len(X_tr),
        "r2": float(r2_score(y_te, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
        "fit_seconds": round(fit_s, 1),
        "predict_seconds": round(pred_s, 2),
    }
    print(name, results[name])

# kernel SVR: document the scaling wall at 10k rows
svr = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10.0))
X_small = X_tr.iloc[:10_000]; y_small = y_tr.iloc[:10_000]
t0 = time.time(); svr.fit(X_small, y_small); fit_s = time.time() - t0
t0 = time.time(); pred = svr.predict(X_te); pred_s = time.time() - t0
results["SVR (RBF, 10k rows only)"] = {
    "train_rows": 10_000,
    "r2": float(r2_score(y_te, pred)),
    "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
    "fit_seconds": round(fit_s, 1),
    "predict_seconds": round(pred_s, 2),
}
print("SVR", results["SVR (RBF, 10k rows only)"])

(ROOT / "report/alternatives.json").write_text(json.dumps(results, indent=2))

# figure: fit time (log) vs test R2
plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#d9d9d9",
                     "grid.linewidth": 0.4, "axes.axisbelow": True,
                     "savefig.dpi": 300})
fig, ax = plt.subplots(figsize=(3.45, 2.6))
colors = {"Linear regression": "#8aa0b4", "k-NN (k=15)": "#8aa0b4",
          "MLP (2x128)": "#8aa0b4", "SVR (RBF, 10k rows only)": "#c0392b",
          "Random Forest": "#1baf7a", "XGBoost": "#2a78d6", "LightGBM": "#eda100"}
for name, r in results.items():
    ax.scatter(r["fit_seconds"], r["r2"], s=26, color=colors[name], zorder=3)
    dy = -0.045 if name in ("XGBoost",) else 0.02
    ax.annotate(name.replace(" (RBF, 10k rows only)", " (10k only)"),
                (r["fit_seconds"], r["r2"]),
                textcoords="offset points", xytext=(4, 3 if dy > 0 else -9),
                fontsize=6.6)
ax.set_xscale("log")
ax.set_xlabel("training time on the common subsample (s, log scale)")
ax.set_ylabel("test R²")
fig.savefig(ROOT / "report/figures/f_alternatives.png",
            bbox_inches="tight", pad_inches=0.02)
print("figure written")
