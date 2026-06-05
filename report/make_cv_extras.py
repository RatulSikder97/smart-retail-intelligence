"""Out-of-fold 5-fold CV for every candidate (class-weighted, no resampling); writes cv_all_models.json and the ROC figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from smart_retail.config import Config
from smart_retail.preprocessing.features import TARGET_COLUMN, feature_matrix
from smart_retail.modeling.classification import build_risk_labels

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
SEED = 42

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "axes.grid": True, "grid.color": "#d9d9d9",
    "grid.linewidth": 0.4, "axes.axisbelow": True, "savefig.dpi": 300,
})

print("loading features…")
features = pd.read_parquet(ROOT / "data/processed/features.parquet")
clusters = pd.read_parquet(ROOT / "data/processed/clusters.parquet")
cluster_labels = pd.Series(
    features[["shop_code"]].merge(clusters, on="shop_code")["cluster"].to_numpy(),
    index=features.index,
)
metrics = json.loads((ROOT / "reports/metrics.json").read_text())
tuned_reg = metrics["phase_3_regression"]["tuning"]["best_params"]
tuned_clf = metrics["phase_4_classification"]["tuning"]["best_params"]

labelled = features[features["has_target"]]
X = feature_matrix(labelled)
y_reg = labelled[TARGET_COLUMN].astype(float)
Xc = X.copy()
Xc["cluster"] = cluster_labels.loc[labelled.index].values
y_clf = build_risk_labels(features, cluster_labels, Config())

out: dict = {"regression": {}, "classification": {}}

# ---------------- regression: 5-fold CV for every candidate ----------------
reg_models = {
    "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=20,
                                           min_samples_leaf=5, n_jobs=-1, random_state=SEED),
    "XGBoost (defaults)": XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.1,
                                       n_jobs=-1, random_state=SEED, verbosity=0),
    "LightGBM": LGBMRegressor(n_estimators=300, num_leaves=63, learning_rate=0.1,
                              n_jobs=-1, random_state=SEED, verbose=-1),
    "XGBoost (tuned)": XGBRegressor(**{**tuned_reg, "tree_method": "hist",
                                       "n_jobs": -1, "random_state": SEED, "verbosity": 0}),
}
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for name, model in reg_models.items():
    r2s, rmses = [], []
    for tr, va in kf.split(X):
        model.fit(X.iloc[tr], y_reg.iloc[tr])
        p = model.predict(X.iloc[va])
        r2s.append(r2_score(y_reg.iloc[va], p))
        rmses.append(float(np.sqrt(mean_squared_error(y_reg.iloc[va], p))))
    out["regression"][name] = {
        "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
        "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
    }
    print("reg", name, out["regression"][name])

# ------------- classification: out-of-fold proba for every candidate -------
pos_weight = float((y_clf == 0).sum() / (y_clf == 1).sum())
clf_models = {
    "Logistic regression": make_pipeline(StandardScaler(),
                                         LogisticRegression(max_iter=2000,
                                                            class_weight="balanced")),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20,
                                            min_samples_leaf=5, class_weight="balanced",
                                            n_jobs=-1, random_state=SEED),
    "XGBoost (defaults)": XGBClassifier(n_estimators=300, max_depth=8,
                                        learning_rate=0.1, scale_pos_weight=pos_weight,
                                        n_jobs=-1, random_state=SEED, verbosity=0),
    "XGBoost (tuned)": XGBClassifier(**{**tuned_clf, "tree_method": "hist",
                                        "scale_pos_weight": pos_weight,
                                        "n_jobs": -1, "random_state": SEED, "verbosity": 0}),
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
roc_points = {}
for name, pipe in clf_models.items():
    proba = cross_val_predict(pipe, Xc, y_clf, cv=skf, method="predict_proba", n_jobs=1)[:, 1]
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_clf, proba)
    keep = np.linspace(0, len(fpr) - 1, 300).astype(int)
    roc_points[name] = (fpr[keep], tpr[keep])
    out["classification"][name] = {
        "roc_auc": float(roc_auc_score(y_clf, proba)),
        "f1": float(f1_score(y_clf, pred)),
        "confusion": confusion_matrix(y_clf, pred).tolist(),
    }
    print("clf", name, out["classification"][name]["roc_auc"])

(ROOT / "report" / "cv_all_models.json").write_text(json.dumps(out, indent=2))

# ROC figure
fig, ax = plt.subplots(figsize=(3.45, 2.9))
colors = {"Logistic regression": "#8aa0b4", "Random Forest": "#1baf7a",
          "XGBoost (defaults)": "#eda100", "XGBoost (tuned)": "#2a78d6"}
for name, (fpr, tpr) in roc_points.items():
    auc = out["classification"][name]["roc_auc"]
    ax.plot(fpr, tpr, lw=1.1, color=colors[name], label=f"{name} (AUC {auc:.3f})")
ax.plot([0, 1], [0, 1], color="#999", lw=0.7, ls="--")
ax.set_xlabel("false positive rate")
ax.set_ylabel("true positive rate")
ax.legend(frameon=False, loc="lower right")
fig.savefig(FIG / "f12_roc.png", bbox_inches="tight", pad_inches=0.02)
print("wrote f12_roc.png")
print("done")
