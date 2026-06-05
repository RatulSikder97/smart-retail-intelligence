# Smart Retail Intelligence Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A fully Dockerized, tested ML pipeline (preprocessing → clustering → sales regression → risk classification → anomaly detection) plus a Streamlit dashboard for Bangladesh's FMCG retail network dataset (719,817 outlets), per the MITE 431 proposal.

**Architecture:** A single Python 3.11 package `smart_retail` with one module per pipeline phase, each exposing a pure `run_*(df, config) -> artifacts` function, orchestrated by a CLI (`python -m smart_retail.pipeline`). Artifacts (parquet + JSON metrics + joblib models) are written to mounted volumes so the Streamlit `dashboard` service reads what the `pipeline` service produced. A synthetic-data generator that matches the real Mendeley schema exactly makes the whole system testable without the 700k-row download.

**Tech Stack:** Python 3.11, pandas, NumPy, scikit-learn, XGBoost, LightGBM, imbalanced-learn, SHAP, Folium, Streamlit, pytest, Docker + docker compose.

## Global Constraints

- Python 3.11 (proposal §6) — Docker base image `python:3.11-slim`.
- Dataset: Mendeley Data `mgzvngzng2` v1, CC BY 4.0, DOI 10.17632/mgzvngzng2.1 (article DOI 10.1016/j.dib.2025.112326).
- Real schema (from the Data in Brief article) is canonical:
  - `shop_info.csv`: shop_code, organizer, place_nature, place_name, freq_type, shop_type, shop_style, shop_importance, ss_importance, visit_target, latitude, longitude, geometry, zila, upazila, division
  - `product_info.csv`: sku_id, sku_name, product_id, product_name, brand, type, packet_size, price_a, price_b, price_c, price_1..3, ingredient, freq, in_packet, total_volume, total_price
  - `product_target_for_shop.csv`: shop_code, prod_1..prod_17, Total
- Bangladesh GPS bounding box for validation: lat 20.5–26.7, lon 88.0–92.7.
- The dataset contains **targets**, not realized sales. RQ2 regresses `Total` target from shop/geo features; RQ3's "will-miss-target" label is a documented proxy: a shop is *at-risk* when its assigned `Total` is < 80% of the median `Total` of its (cluster × shop_type) peer group — i.e., materially under-performing its peers' expectation.
- Every phase must run on the synthetic sample (~5,000 shops) in < 2 min so CI/tests stay fast; row-count knobs via config.
- No TODOs/placeholders; pytest green + `docker compose build` green before done.

## File Structure

```
SMART_RETAIL/
├── Dockerfile                     # single image, shared by both services
├── docker-compose.yml             # services: pipeline, dashboard, test
├── .dockerignore / .gitignore / Makefile / README.md
├── pyproject.toml                 # package + deps + pytest config
├── docs/plans/                    # this plan
├── data/raw/                      # CSVs land here (volume-mounted)
├── data/processed/                # parquet artifacts per phase
├── models/                        # joblib models
├── reports/                       # metrics JSON, cluster profiles, figures
├── scripts/
│   ├── download_data.py           # Mendeley public API downloader
│   └── generate_sample_data.py    # synthetic data, exact real schema
├── src/smart_retail/
│   ├── __init__.py
│   ├── config.py                  # paths + all tunables (dataclass)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── schema.py              # column constants, dtypes, BD bbox
│   │   └── loader.py              # load + validate the 3 CSVs
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py         # Phase 1: clean, densities, distances
│   ├── models/
│   │   ├── __init__.py
│   │   ├── clustering.py          # Phase 2 (RQ1)
│   │   ├── regression.py          # Phase 3 (RQ2)
│   │   ├── classification.py      # Phase 4 (RQ3)
│   │   └── anomaly.py             # Phase 5 (RQ4)
│   └── pipeline.py                # CLI orchestrator (argparse)
├── dashboard/
│   └── app.py                     # Streamlit, 5 tabs
└── tests/
    ├── conftest.py                # synthetic fixtures (small, seeded)
    ├── test_schema_loader.py
    ├── test_features.py
    ├── test_clustering.py
    ├── test_regression.py
    ├── test_classification.py
    ├── test_anomaly.py
    └── test_pipeline.py           # end-to-end on tiny sample
```

---

### Task 1: Scaffold + config

**Files:** Create `pyproject.toml`, `.gitignore`, `Makefile`, `README.md`, `src/smart_retail/__init__.py`, `src/smart_retail/config.py`.

**Produces:** `Config` dataclass with `data_dir`, `raw_dir`, `processed_dir`, `models_dir`, `reports_dir`, phase tunables (`kmeans_k_range=(3, 9)`, `density_radii_km=(0.5, 1.0, 5.0)`, `risk_threshold=0.8`, `anomaly_top_decile=0.1`, `random_state=42`), and `Config.from_env()` reading `SMART_RETAIL_DATA_DIR` etc. so Docker volumes override paths.

- [ ] pyproject with deps: pandas, numpy, scikit-learn, xgboost, lightgbm, imbalanced-learn, shap, folium, streamlit, streamlit-folium, pyarrow, joblib, requests, tqdm; dev: pytest.
- [ ] `git init`, initial commit.

### Task 2: Docker environment

**Files:** Create `Dockerfile`, `docker-compose.yml`, `.dockerignore`.

- [ ] `Dockerfile`: `python:3.11-slim`, install `libgomp1` (LightGBM/XGBoost OpenMP), copy `pyproject.toml` + `src/` and `pip install -e .`, non-root user.
- [ ] `docker-compose.yml`: `pipeline` (runs `python -m smart_retail.pipeline --sample` by default), `dashboard` (streamlit on 8501), `test` (pytest) — all sharing the image and `./data`, `./models`, `./reports` volumes.
- [ ] Verify: `docker compose build` exits 0.

### Task 3: Data layer

**Files:** Create `src/smart_retail/data/schema.py`, `src/smart_retail/data/loader.py`, `scripts/generate_sample_data.py`, `scripts/download_data.py`, `tests/conftest.py`, `tests/test_schema_loader.py`.

**Produces:**
- `schema.py`: `SHOP_COLUMNS`, `PRODUCT_COLUMNS`, `TARGET_COLUMNS` (prod_1..prod_17 + Total), `BD_BBOX = (20.5, 26.7, 88.0, 92.7)`, `DIVISIONS` (8 real names).
- `loader.load_shops(config) -> DataFrame`, `load_products(config)`, `load_targets(config)`; raise `FileNotFoundError` with download hint; validate required columns.
- `generate_sample_data.main(n_shops, out_dir, seed)`: seeded synthetic CSVs, exact schema, realistic value distributions incl. ~10% missing organizer/shop_type, ~13% missing GPS, duplicate shop_codes, out-of-bbox GPS noise — so cleaning code has something real to do.
- `download_data.py`: Mendeley public API (`https://data.mendeley.com/public-api/datasets/mgzvngzng2/files?folder_id=root&version=1`), streams each file to `data/raw/`.

- [ ] Tests: generated sample matches schema; loader round-trips it; loader raises on missing file.

### Task 4: Phase 1 — preprocessing & feature engineering

**Files:** Create `src/smart_retail/features/engineering.py`, `tests/test_features.py`.

**Produces:** `run_preprocessing(shops, targets, config) -> (features_df, quality_report_dict)` writing `data/processed/features.parquet` when invoked via pipeline. Steps: drop duplicate shop_codes, flag/filter invalid GPS (outside bbox or null), fill categorical NAs with "Unknown", per-shop features: `density_500m`, `density_1km`, `density_5km` (BallTree haversine), `dist_to_district_centre_km` (centroid of zila's shops), one-hot/frequency-encode shop_type, shop_style, shop_importance, division; join `Total` target + per-product breadth (`n_products_targeted`).

- [ ] Tests: duplicates removed, bbox filtering works, densities monotonic (500m ≤ 1km ≤ 5km), feature matrix has no NaNs.

### Task 5: Phase 2 — clustering (RQ1)

**Files:** Create `src/smart_retail/models/clustering.py`, `tests/test_clustering.py`.

**Produces:** `run_clustering(features_df, config) -> ClusteringResult` (dataclass: `labels`, `best_k`, `metrics` {silhouette, davies_bouldin per k per algo}, `profiles` DataFrame). K-Means over k range scored by silhouette + Davies–Bouldin; Agglomerative (on ≤20k sample) and DBSCAN for comparison; standard-scaled geo+structural features; profile = per-cluster medians of key features.

- [ ] Tests: labels cover all rows, best_k in range, profiles has best_k rows, deterministic under seed.

### Task 6: Phase 3 — regression (RQ2)

**Files:** Create `src/smart_retail/models/regression.py`, `tests/test_regression.py`.

**Produces:** `run_regression(features_df, config) -> RegressionResult` (`metrics` per model {rmse, mae, r2}, `best_model_name`, fitted `best_model`, `shap_importance` DataFrame, `predictions` Series aligned to shop_code). Models: LinearRegression baseline, RandomForest, XGBoost, LightGBM; train/test split 80/20; SHAP TreeExplainer on best tree model (subsampled ≤2,000 rows).

- [ ] Tests: all 4 models produce finite metrics, tree models beat-or-match baseline on synthetic signal, shap_importance non-empty.

### Task 7: Phase 4 — classification (RQ3)

**Files:** Create `src/smart_retail/models/classification.py`, `tests/test_classification.py`.

**Produces:** `run_classification(features_df, cluster_labels, config) -> ClassificationResult` (`labels` proxy definition string, `metrics` per model {roc_auc, f1, precision, recall}, `best_model`, `risk_scores` Series). Label: at-risk if `Total < risk_threshold × median(Total | cluster, shop_type)`. SMOTE on training folds only; LogisticRegression, RandomForest, XGBoost.

- [ ] Tests: label rate strictly between 0 and 1, metrics finite, no SMOTE leakage (fit on train only — assert test set untouched), risk_scores in [0,1].

### Task 8: Phase 5 — anomaly detection (RQ4)

**Files:** Create `src/smart_retail/models/anomaly.py`, `tests/test_anomaly.py`.

**Produces:** `run_anomaly(features_df, cluster_labels, config) -> AnomalyResult` (`scores` DataFrame with iforest_score, lof_score, combined rank, `top_anomalies` top-decile DataFrame with per-shop context). IsolationForest + LOF fitted *within each cluster*; combined score = mean of per-method percentile ranks.

- [ ] Tests: every shop scored, top_anomalies ≈ top decile size, per-cluster fitting (clusters with <20 shops skipped gracefully).

### Task 9: Pipeline orchestrator

**Files:** Create `src/smart_retail/pipeline.py`, `tests/test_pipeline.py`.

**Produces:** `python -m smart_retail.pipeline [--sample N] [--phases 1,2,3,4,5] [--data-dir PATH]`. Runs phases sequentially, persists: `features.parquet`, `clusters.parquet`, `predictions.parquet`, `risk.parquet`, `anomalies.parquet`, `reports/metrics.json` (all phases' metrics + timestamps), `models/*.joblib`. `--sample N` generates synthetic data first if `data/raw` is empty.

- [ ] Tests: end-to-end on 800 synthetic shops produces all artifacts; metrics.json parses.

### Task 10: Streamlit dashboard

**Files:** Create `dashboard/app.py`.

**Produces:** 5 tabs reading artifacts via `Config.from_env()`: **Overview** (KPIs + folium map sample colored by cluster), **Archetypes** (cluster profiles + division breakdown), **Sales Prediction** (model leaderboard, SHAP bar chart, actual-vs-predicted scatter), **Risk** (risk-rate by division/cluster, top at-risk table), **Anomalies** (ranked table + map of top anomalies). Graceful empty-state when artifacts missing ("run the pipeline first" + command).

### Task 11: E2E Docker verification

- [ ] `docker compose build`
- [ ] `docker compose run --rm test` → pytest green
- [ ] `docker compose run --rm pipeline` (sample mode) → artifacts exist
- [ ] `docker compose up -d dashboard` → HTTP 200 on :8501, then down.

## Self-Review Notes

- Spec coverage: RQ1→Task 5, RQ2→Task 6, RQ3→Task 7, RQ4→Task 8, Phase 1→Task 4, dashboard deliverable→Task 10, Docker request→Task 2, dataset access→Task 3. Final report (proposal deliverable f) is out of scope for the codebase; README documents results locations.
- Known judgment call: "will-miss-target" proxy label (no realized-sales column exists) — documented in Global Constraints and README.
- GeoPandas from the proposal toollist is intentionally dropped: BallTree haversine + folium cover the geo needs without the heavy GDAL stack in Docker.
