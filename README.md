# Smart Retail Intelligence for Bangladesh's FMCG Distribution Network

An integrated **supervised, unsupervised, and clustering** pipeline over Bangladesh's
national retail dataset (719,817 GPS-tagged outlets), with a Streamlit dashboard.

Course: **MITE 431 — Big Data Analytics** · Ratul Sikder (2506102) & Fahim Rahman (2506108)

## Research questions → pipeline phases

| Phase | RQ | What it does | Models |
|---|---|---|---|
| 1. Preprocessing | — | GPS validation, dedup, shop-density & distance features | — |
| 2. Clustering | RQ1 | Shop archetypes across 8 divisions | K-Means, Hierarchical, DBSCAN |
| 3. Regression | RQ2 | Predict monthly sales target from location/type/density | Linear, RF, XGBoost, LightGBM |
| 4. Classification | RQ3 | Predict which shops will miss targets (proxy label) | LogReg, RF, XGBoost + SMOTE |
| 5. Anomaly detection | RQ4 | Per-cluster outliers (fraud / errors / opportunity) | Isolation Forest, LOF |

## Dataset

[Comprehensive Retail Network Dataset of Bangladesh](https://data.mendeley.com/datasets/mgzvngzng2/1)
(Shahin, Swarna & Farid, 2025) — CC BY 4.0, DOI `10.17632/mgzvngzng2.1`.

Download the three CSVs from Mendeley and place them in `dataset/`:

```
dataset/
├── shop_info_with_geo.csv        (~117 MB, 719,817 shops)
├── product_target_for_shop.csv   (~89 MB, 613,656 shops × 17 products)
└── product_info.csv              (~4 KB, 42 SKUs)
```

> **Note:** the dataset ships sales **targets**, not realized sales. The RQ3
> "will-miss-target" label is therefore a documented proxy: a shop is *at-risk*
> when its total target is below 80% of the median of its (cluster × shop type)
> peer group.

## Quick start (Docker)

```bash
make build        # build the image
make pipeline     # run phases 1-5 → data/processed/, models/, reports/
make dashboard    # Streamlit on http://localhost:8501
make test         # pytest inside the container
```

Or without Make:

```bash
docker compose build
docker compose run --rm pipeline
docker compose up dashboard
```

## Big-data layer (Spark)

A Spark 3.5 standalone cluster ships in the same compose file for the
distributed-processing part of the course:

```bash
make spark-up     # spark-master (UI http://localhost:8080) + spark-worker (:8081)
make spark-etl    # distributed ETL job over the raw CSVs → data/spark/
make spark-down
```

`spark_jobs/etl_preprocess.py` re-expresses the phase-1 cleaning rules as
distributed Spark SQL (dedup, GPS validation, valued-target window join) and
writes `data/spark/clean_shops.parquet` plus per-division analytics. It
reproduces the pandas pipeline's counts exactly (832,881 → 719,817 → 613,660)
— a cross-validation of both implementations.

### Scaling out: regional partitioning + dynamic topology

```bash
WORKERS=2 DATANODES=9 make spark-up   # one datanode per data partition:
                                      # 8 divisions + the 'Unknown' bucket
make spark-partition                  # division-partitioned warehouse in HDFS
DIVISIONS=Khulna,Barisal make spark-pipeline   # regional (pruned) MLlib run
make spark-pipeline                   # full national run
```

Design choice — **partition by division, scale datanodes freely** (rather than
"one datanode per district"): HDFS distributes *blocks*, not semantic regions,
and 64 district shards of a 206 MB dataset would average ~3 MB against a
128 MB block size (the classic small-files problem). Instead the warehouse is
written Hive-style (`…/warehouse/shops/division=Khulna/…`, 8+1 partitions), so
any regional job engages Spark partition pruning — the physical plan carries
`PartitionFilters: [division = Khulna]` and never reads other divisions'
files. Datanodes and Spark workers scale horizontally via `deploy.replicas`
(`WORKERS`/`DATANODES` env), and raw data is stored with replication 2 so the
loss of any single datanode loses nothing.

Operational notes: HDFS metadata and blocks live inside the containers (demo
setup) — after recreating the namenode or datanodes, re-run the ingest. Job
targets use `--no-deps` so running a job never rescales the live topology.

### Every exposed endpoint

| Endpoint | URL | What to check there |
|---|---|---|
| Dashboard | http://localhost:8501 | all analysis views |
| HDFS namenode UI | http://localhost:9870 | datanodes, blocks, browse `/smart_retail` |
| HDFS RPC | `hdfs://localhost:8020` | CLI/API access from the host |
| Spark master UI | http://localhost:8080 | workers, running/completed jobs |
| Spark REST submit | http://localhost:6066 | programmatic job submission |
| Spark driver UI | http://localhost:4040 | live stages/DAG while a job runs |
| Datanode UIs (×N) | auto-assigned host ports | per-node blocks and health |
| Worker UIs (×N) | auto-assigned host ports | per-worker executors |

Replicas can't share one host port, so datanode/worker UIs get auto-assigned
ports — `make spark-ports` prints the live map.

**Where is Hadoop?** Spark bundles the Hadoop 3 client libraries (this build
is `spark-3.5-bin-hadoop3`), which is how it reads/writes the shared volume.
A full HDFS + YARN cluster adds nothing at 206 MB of input — the mounted
volume plays the storage role — and the classic Hadoop Docker images are
amd64-only (slow emulation on Apple Silicon). If the course requires visible
HDFS daemons, that can be added as another compose profile.

## Local development (no Docker)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m smart_retail.pipeline          # full pipeline
python -m smart_retail.pipeline --phases 1,2   # selected phases
pytest
streamlit run dashboard/app.py
```

## Outputs

| Artifact | Path |
|---|---|
| Engineered features | `data/processed/features.parquet` |
| Cluster labels + profiles | `data/processed/clusters.parquet`, `reports/cluster_profiles.csv` |
| Sales predictions | `data/processed/predictions.parquet` |
| Risk scores | `data/processed/risk.parquet` |
| Ranked anomalies | `data/processed/anomalies.parquet` |
| All metrics | `reports/metrics.json` |
| Trained models | `models/*.joblib` |

## Project layout

```
src/smart_retail/
├── config.py               # paths + tunables (env-overridable)
├── data/                   # schema constants, loaders
├── features/engineering.py # phase 1
├── models/                 # phases 2-5 (clustering, regression, classification, anomaly)
└── pipeline.py             # CLI orchestrator
dashboard/app.py            # Streamlit (5 tabs)
tests/                      # pytest suite (runs on real-data slices)
```
