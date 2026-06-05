"""Evidence for the imbalance-handling choice (RQ3).

Same XGBoost, same stratified 80/20 split, four balancing strategies:
none, class weights (scale_pos_weight), random undersampling, SMOTE.
Results land in report/imbalance.json for the report and the dashboard.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from smart_retail.config import Config
from smart_retail.preprocessing.features import feature_matrix
from smart_retail.modeling.classification import build_risk_labels

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

features = pd.read_parquet(ROOT / "data/processed/features.parquet")
clusters = pd.read_parquet(ROOT / "data/processed/clusters.parquet")
cluster_labels = pd.Series(
    features[["shop_code"]].merge(clusters, on="shop_code")["cluster"].to_numpy(),
    index=features.index,
)
labelled = features[features["has_target"]]
X = feature_matrix(labelled)
X["cluster"] = cluster_labels.loc[labelled.index].values
y = build_risk_labels(features, cluster_labels, Config())

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=SEED, stratify=y)
pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())

def xgb(**extra):
    return XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                         n_jobs=-1, random_state=SEED, verbosity=0, **extra)

strategies = {
    "No balancing": xgb(),
    "Class weights (scale_pos_weight)": xgb(scale_pos_weight=pos_weight),
    "Random undersampling": ImbPipeline([("rus", RandomUnderSampler(random_state=SEED)),
                                         ("m", xgb())]),
    "SMOTE (ours)": ImbPipeline([("sm", SMOTE(random_state=SEED)), ("m", xgb())]),
}

results = {}
for name, model in strategies.items():
    t0 = time.time()
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    results[name] = {
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "f1": float(f1_score(y_te, pred)),
        "recall": float(((pred == 1) & (y_te == 1)).sum() / (y_te == 1).sum()),
        "fit_seconds": round(time.time() - t0, 1),
    }
    print(name, results[name])

(ROOT / "report/imbalance.json").write_text(json.dumps(results, indent=2))
print("written report/imbalance.json")
