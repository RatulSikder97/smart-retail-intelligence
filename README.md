# Smart Retail Intelligence for Bangladesh's FMCG Distribution Network

Term project for MITE 431 (Big Data Analytics), Institute of Information Technology, University of Dhaka — Ratul Sikder (2506102) and Fahim Rahman (2506108).

We work on the national retail dataset of Bangladesh: 719,817 GPS-tagged shops, mostly the small mudi shops and general stores you find on every street here. On this data we answer four research questions — what shop archetypes exist (clustering), can we predict a shop's monthly sales target (regression), which shops are at risk of missing their target (classification), and which shops look anomalous inside their own cluster (anomaly detection). The same five-phase pipeline runs in two lanes: a standard single-machine run with scikit-learn, and a distributed run on Hadoop HDFS plus Spark MLlib.

## What you need

- Docker Desktop. Everything runs in containers, so no local Python setup is needed.
- The dataset. Download the three CSVs from [Mendeley Data](https://data.mendeley.com/datasets/mgzvngzng2/1) and place them in `dataset/`:

```
dataset/
├── shop_info_with_geo.csv        (~117 MB, 719,817 shops)
├── product_target_for_shop.csv   (~89 MB, 613,656 shops × 17 products)
└── product_info.csv              (~4 KB, 42 SKUs)
```

## Run the standard pipeline (single machine)

1. `make build` — builds the Docker image. The first build takes a few minutes while pip pulls the dependencies.
2. `make pipeline` — runs all five phases (preprocessing, clustering, regression, classification, anomaly detection). It writes cleaned data to `data/processed/`, trained models to `models/`, and every metric to `reports/`. On our machines it takes about 6 minutes end to end.
3. `make dashboard` — serves the Streamlit dashboard at http://localhost:8501. Run this after the pipeline so it has results to show. Stop it with Ctrl+C.
4. `make test` — runs the pytest suite inside the container. We use this to check nothing broke after a change.

## Run the distributed pipeline (Hadoop + Spark)

This lane repeats the whole analysis on a containerised cluster. We ran most of our experiments with 2 Spark workers and 9 HDFS datanodes (one per division partition plus the Unknown bucket).

1. `WORKERS=2 DATANODES=9 make spark-up` — starts the namenode, the datanodes, the Spark master, and the workers in the background. Give the namenode a moment to come up.
2. Ingest the raw CSVs into HDFS:

   ```
   docker compose --profile spark --profile spark-jobs run --rm --no-deps hdfs-ingest
   ```

   We upload through a small Spark job instead of the hadoop CLI because the emulated CLI kept aborting on the large files.
3. `make spark-partition` — builds the division-partitioned Parquet warehouse in HDFS.
4. `make spark-pipeline` — runs the five phases with Spark MLlib across the cluster. For a regional run, name the divisions: `DIVISIONS=Khulna,Barisal make spark-pipeline`. Spark then prunes the other divisions' partitions and never reads them.
5. `make spark-down` — stops the whole cluster when you are done.

While the cluster is up you can watch it work: the namenode web UI is at http://localhost:9870 and the Spark master UI at http://localhost:8080 (a live driver UI also appears on port 4040 during a job). `make spark-ports` prints every exposed port.

## Optional extras

- Hyperparameter tuning and cross-validation on the supervised phases:

  ```
  docker compose run --rm pipeline python -m smart_retail.pipeline --phases 3,4 --tune --cv
  ```

  `--tune` runs the Optuna search and `--cv` adds 5-fold cross-validation of the best model. This is slow, so we kept it out of the default run.
- The k-NN imputation sensitivity experiment: add `--impute-knn` to the same command. By default we keep missing categories as an explicit `Unknown` level, because the shops with missing fields turned out to be a signal in themselves. The flag fills geo fields and high-confidence categoricals instead, marks every filled value with a flag column, and writes the comparison to `reports/imputation_sensitivity.json`.
- Regenerating the report figures after a full pipeline run: `report/make_figures.py` (along with `report/make_cv_extras.py` and `report/make_alternatives.py`) rebuilds the PNGs in `report/figures/` from the saved metrics.

## Project layout

```
src/smart_retail/   standard pipeline: config, phases 1-5, tuning, reporting
spark_jobs/         distributed lane: HDFS upload, ETL, partitioning, MLlib pipeline
dashboard/          Streamlit app
dataset/            the three raw CSVs (you place these; mounted read-only)
data/               processed outputs written by the pipeline
models/             saved model artifacts
reports/            metrics, profiles, and JSON outputs from every run
report/             the written report, its figures, and the figure scripts
tests/              pytest suite
```

Every number quoted in the written report comes from `reports/metrics.json`, so you can re-run the pipeline and verify all of them yourself.
