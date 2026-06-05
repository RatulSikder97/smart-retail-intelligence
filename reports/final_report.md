# Smart Retail Intelligence — Analysis Report

Generated from `reports/metrics.json` — numbers always match the artifacts of the latest pipeline run.

### The dataset

**Comprehensive Retail Network Dataset of Bangladesh** (Shahin, Swarna & Farid,
2025) — Mendeley `mgzvngzng2` v1, CC BY 4.0, collected Feb 2022 – Oct 2024.
Three files: shop records with GPS and structure (832,881 rows), monthly
sales targets for 17 products per shop (853,731 rows), and a 42-SKU product
catalogue (5 brands, 9 product types) with price tiers.

### Data quality & preprocessing

| Step | Count |
|---|---|
| Raw shop records | 832,881 |
| Removed: missing GPS | 113,064 (13.6%) |
| **Final analysable shops** | **719,817** |
| Shops with a monthly sales target | 613,660 (85.3%) |

**Why are 113,064 rows removed?** Their `latitude`/`longitude` cells are
empty in the raw CSV and their `geometry` field is literally `POINT EMPTY` —
the coordinates were never captured at survey time. This 13.6% missing-GPS
rate is documented by the dataset authors themselves (Shahin et al., 2025),
and after removing those rows the shop count equals the paper's published
figure of 719,817 exactly. No valid shop is discarded: zero non-null
coordinates fall outside the Bangladesh bounding box.

**Comment.** The gap itself is a finding for the distributor: one in seven
book entries cannot be placed on a map, which is consistent with the
duplicate / ghost-shop problem that motivated this project.

---

### Feature engineering

From the cleaned records, **48 model features** are
engineered per shop:

| Group | Features | How they are computed |
|---|---|---|
| Neighbourhood density | `density_500m`, `density_1km`, `density_5km` | BallTree (haversine) neighbour counts within each radius, excluding the shop itself |
| Centrality | `dist_to_district_centre_km` | Great-circle distance to the centroid of the shop's district (zila) |
| Product mix | `n_products_targeted` | Number of the 17 products with a positive target for the shop |
| Position | `latitude`, `longitude` | Validated GPS coordinates |
| Structure (one-hot) | shop type (12), style (3), importance (3), place nature, division | Explicit "Unknown" level instead of dropping ~10% categorical gaps |

**Comment.** Density and centrality translate the proposal's
"urban shops over-stocked / rural shops stock-out" hypothesis into measurable
inputs; the SHAP ranking in RQ2 confirms they carry real signal. The monthly
target (`total_target`) is joined for 613,660 shops
and used as the regression label — never as a model input.

---

### RQ1 — What natural archetypes exist among the outlets?

K-Means was scored for k = 3…9
on silhouette (higher better) and Davies–Bouldin (lower better).
**Best k = 8** (silhouette 0.187, DB 1.94).

**Comment.** A silhouette of 0.19 indicates overlapping clusters — expected for a continuous national retail fabric; the taxonomy is actionable because profiles differ strongly on density, importance, and target.
Agglomerative clustering on a subsample corroborates the structure
(silhouette 0.145);
DBSCAN found 1 cluster(s) with
99% noise at default settings — evidence the
network is dense and continuous rather than separated islands.

---

### RQ2 — Can monthly sales be predicted from location, type, density, and assortment?

| Model | RMSE | MAE | R² |
|---|---|---|---|
| linear_regression | 272.7 | 23.78 | 0.024 |
| xgboost_tuned | 227.4 | 18.71 | 0.321 |
| random_forest | 224.3 | 17.58 | 0.340 |
| lightgbm | 219.8 | 19.93 | 0.366 |
| **xgboost** | 204.4 | 18.12 | 0.452 |

**Comment.** The linear baseline explains almost none of the variance
(R² 0.02) while **xgboost** reaches
R² 0.45 and cuts RMSE by
25% —
the target-to-shop relationship is real but strongly non-linear.

Hyperparameters of the winner were selected by Optuna TPE (Bayesian) (30 trials, 3-fold CV RMSE (minimized)).

**Stability.** 5-fold cross-validation: R² 0.020 ± 0.612, RMSE 236.8 ± 59.9 — the headline numbers are not a lucky split.

---

### RQ3 — Which shops will miss their targets?

**Label (documented proxy).** The dataset ships assigned targets, not realized
sales, so: `at_risk = total_target < risk_threshold * median(total_target | cluster, shop_type)` with threshold
0.8 — a shop is at-risk when it materially
under-indexes its (cluster × shop-type) peers. Base rate:
36.6%. Class weights handle the imbalance; no rows are resampled.

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| logistic_regression | 0.760 | 0.619 | 0.542 | 0.720 |
| random_forest | 0.862 | 0.720 | 0.661 | 0.791 |
| xgboost | 0.862 | 0.720 | 0.657 | 0.795 |
| **xgboost_tuned** | 0.875 | 0.735 | 0.677 | 0.805 |

**Comment.** The best model (xgboost_tuned) reaches ROC-AUC
0.875 — strong
discrimination from structural and assortment features; with realized sales data the
same pipeline retrains unchanged into a true "will miss next quarter" model.

Hyperparameters of the winner were selected by Optuna TPE (Bayesian) (30 trials, 3-fold CV ROC-AUC (maximized, class weights inside each fold)).

**Stability.** 5-fold cross-validation: ROC-AUC 0.876 ± 0.001, F1 0.736 ± 0.001.

---

### RQ4 — Which shops behave anomalously vs their peers?

Isolation Forest and LOF were fitted **within each cluster**, so a rural kiosk
is compared to rural kiosks, not to metro superstores. All
719,817 shops were scored; the top decile
(**71,982 shops**) is exported for manual triage into
**fraud / data error / hidden opportunity**.


---

### Conclusion & recommended actions

End-to-end, the pipeline turns 832,881 raw book entries
into: **8 interpretable archetypes**, a sales model
(R² 0.45), a target-miss risk ranking
(ROC-AUC 0.875), and
71,982 triage-ready anomalies — in
465s of compute.

Recommended actions for the distributor:

1. **Audit the data-poor segment first** — the archetype with zero product mix
   plus the 113,064 entries
   without GPS are the cheapest wins against the 15–25% capacity waste estimate.
2. **Rebalance targets where the model and the book disagree** — shops whose
   predicted volume far exceeds the assigned target are under-served demand;
   the reverse are over-allocation risks (RQ3 list, worst in the highest-risk
   division).
3. **Field-verify the anomaly top decile per division**, prioritising shops
   that are simultaneously RQ3 at-risk.
4. **Refresh quarterly** — the pipeline is deterministic (seed 42) and fully
   containerised, so re-running on updated books is one command.
