# Smart Retail Intelligence for Bangladesh's FMCG Distribution Network
### Final Report — MITE 431, Big Data Analytics
Ratul Sikder (2506102) & Fahim Rahman (2506108) · July 2026

---

## 1. Summary

We built an integrated supervised, unsupervised, and clustering pipeline over the
Comprehensive Retail Network Dataset of Bangladesh (Shahin, Swarna & Farid, 2025;
Mendeley `mgzvngzng2` v1, CC BY 4.0) and answered all four research questions.
After GPS validation and deduplication, **719,817 outlets** across all 8 divisions
and 64 districts were analysed — matching the count published with the dataset.
The full five-phase pipeline runs end-to-end in Docker in roughly **3 minutes**,
and an interactive Streamlit dashboard integrates all four analytical layers.

All numbers below come from `reports/metrics.json` and `reports/cluster_profiles.csv`,
produced by `python -m smart_retail.pipeline` on the full dataset.

## 2. Data & preprocessing (Phase 1)

| Step | Count |
|---|---|
| Raw shop records | 832,881 |
| Missing or out-of-country GPS removed | 113,064 (13.6%) |
| Final analysable shops | **719,817** |
| Shops with a monthly sales target | 613,660 (85.3%) |
| Engineered model features | 48 |

Engineered features: neighbour density within 500 m / 1 km / 5 km (haversine
BallTree), distance to the shop's district centroid, product-mix breadth
(`n_products_targeted`), and one-hot structural attributes (shop type, style,
importance, place nature, division). Categorical gaps (~10%) were kept as an
explicit "Unknown" level rather than dropped.

**Data quality findings for the distributor:** 2.4% of shop codes in the targets
file are duplicated (sometimes with conflicting rows), ~14% of outlets lack GPS,
and roughly 15% of shops carry no target at all — direct evidence of the
"duplicates, ghost shops" problem that motivated the project.

## 3. RQ1 — Shop archetypes (K-Means, Hierarchical, DBSCAN)

K-Means was scored for k = 3…9 on silhouette and Davies–Bouldin over the scaled
48-feature matrix. **k = 8 won** (silhouette 0.187, DB 1.94, the best of both
criteria jointly); agglomerative clustering on a 20k subsample corroborated the
structure (silhouette 0.145), while DBSCAN at default settings collapsed to one
cluster with 98.7% noise — evidence the network is dense and continuous rather
than separated islands, which is itself a finding.

The eight archetypes are interpretable and map cleanly onto geography:

| # | Share | Profile (medians) | Reading |
|---|---|---|---|
| 4 | 28.5% | Dhaka-belt periphery, density 66/km, target 15.2 | The typical semi-urban shop |
| 0 | 19.0% | Rajshahi/north-west, low density, target 11.1 | Rural north-west backbone |
| 6 | 15.5% | Chittagong coast, density 91/km, target 10.5 | South-eastern corridor |
| 3 | 12.3% | **Metro Dhaka core** — density 869/km (10,484 within 5 km), 8.2 km from centre, target 25.4 | Urban high-throughput stores |
| 1 | 11.8% | *No product mix, unknown type/importance* | **Data-poor "ghost-shop" segment** |
| 5 | 6.4% | Mymensingh, most remote (18.9 km) | Deep-rural cluster |
| 7 | 5.0% | Sylhet region, low density | North-eastern pocket |
| 2 | 1.4% | High-importance wholesale-style, median target **90.0** (6–8× the rest) | Key-account outlets |

Cluster 1 (85k shops with empty categorical data and zero targeted products) is a
machine-found duplicate/ghost candidate list; cluster 2 isolates the small
key-account tier that carries disproportionate volume.

## 4. RQ2 — Sales-target regression

80/20 train/test split, 490,928 training shops. The linear baseline confirms the
problem is non-linear:

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear regression (baseline) | 272.7 | 23.78 | 0.02 |
| Random Forest | 224.3 | 17.58 | 0.34 |
| LightGBM | 219.8 | 19.93 | 0.37 |
| **XGBoost** | **204.4** | 18.12 | **0.45** |

SHAP attribution on the XGBoost winner ranks the drivers: product-mix breadth
(`n_products_targeted`, 17.1) and top-tier importance flags (16.6) dominate,
followed by micro-location — density within 500 m (6.4), 5 km (5.2), raw
latitude/longitude (~4.8 each), and distance to district centre (3.0). Location
and assortment, not shop type alone, drive expected volume — supporting the
proposal's premise that allocation by intuition leaves signal on the table.

## 5. RQ3 — Target-miss risk classification

Because the dataset ships assigned targets rather than realized sales, the label
is a documented proxy: a shop is **at-risk** when its total target is below 80%
of the median of its (cluster × shop-type) peer group — i.e. materially
under-indexing shops that look like it. Base rate: 36.6%. SMOTE was applied
inside the imblearn pipeline (training folds only).

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic regression | 0.760 | 0.618 | 0.542 | 0.720 |
| Random Forest | 0.844 | 0.682 | 0.699 | 0.667 |
| **XGBoost** | **0.857** | **0.695** | 0.716 | 0.675 |

Risk is geographically uneven: predicted at-risk share ranges from 27.8% in
Barisal to **48.5% in Khulna**, with Rangpur second at 40.6% — a concrete
prioritisation map for distributor field teams.

## 6. RQ4 — Anomaly detection

Isolation Forest and LOF were fitted **within each of the 8 clusters** (so shops
are compared to their own archetype), combined by mean percentile rank. All
719,817 shops were scored; the **top decile (71,982 shops)** is exported with
full context (`data/processed/top_anomalies.parquet`) for manual triage into
fraud / data error / hidden opportunity, as proposed.

## 7. Deliverables

| Proposal deliverable | Where |
|---|---|
| (a) Retail taxonomy, 8 archetypes | `reports/cluster_profiles.csv`, dashboard "Shop archetypes" |
| (b) Benchmarked sales model + SHAP | `models/regression_best.joblib`, `reports/shap_importance.csv` |
| (c) Target-miss risk classifier | `models/classification_best.joblib`, `data/processed/risk.parquet` |
| (d) Ranked anomaly list | `data/processed/top_anomalies.parquet` |
| (e) Interactive dashboard | `make dashboard` → http://localhost:8501 |
| (f) Final report | this document |

Reproducibility: `make build && make pipeline` (Docker, seed 42 throughout);
33 automated tests (`make test`) run against slices of the real dataset.

## 8. Limitations & future work

- The RQ3 label is a peer-relative proxy; with realized monthly sales the same
  pipeline retrains unchanged into a true "will miss next quarter" model.
- Silhouette ≈ 0.19 indicates overlapping (not sharply separated) archetypes —
  expected for a continuous national retail fabric; the taxonomy is still
  actionable because profiles differ strongly on density, importance, and target.
- Hyperparameters are sensible fixed choices; Bayesian tuning (as proposed) is a
  drop-in upgrade via the config, at ~10× compute cost.
- DBSCAN would need per-cluster eps tuning to be informative at this density.
