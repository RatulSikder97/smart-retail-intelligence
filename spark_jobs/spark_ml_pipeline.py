"""Distributed 5-phase pipeline on Hadoop (HDFS) + Spark MLlib, mirroring the scikit-learn reference pipeline."""

import json
import os
import sys
import time

from pyspark.ml.clustering import BisectingKMeans, KMeans
from pyspark.ml.classification import (
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    ClusteringEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
)
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

HDFS = "hdfs://namenode:8020/smart_retail"
LOCAL_METRICS = "/data/out/spark/spark_metrics.json"

LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 20.5, 26.7, 88.0, 92.7
PRODUCT_COLS = [f"prod_{i}" for i in range(1, 18)]
DENSITY_RADII_KM = (0.5, 1.0, 5.0)
KM_PER_DEG_LAT = 110.6
KM_PER_DEG_LON = 101.9  # at ~23.7°N
K_RANGE = range(3, 10)
SEED = 42
RISK_THRESHOLD = 0.8

ONEHOT_SOURCE = ["shop_type", "shop_style", "shop_importance", "place_nature", "division"]


def grid_density(df, radius_km: float, label: str):
    # neighbour count via 3x3 grid-cell window (cell edge = radius) — avoids the O(n²) pairwise join
    cx = F.floor(F.col("longitude") / (radius_km / KM_PER_DEG_LON))
    cy = F.floor(F.col("latitude") / (radius_km / KM_PER_DEG_LAT))
    with_cell = df.withColumn("_cx", cx).withColumn("_cy", cy)
    cell_counts = with_cell.groupBy("_cx", "_cy").count()
    offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    window_counts = None
    for dx, dy in offsets:
        shifted = cell_counts.select(
            (F.col("_cx") + dx).alias("_cx"),
            (F.col("_cy") + dy).alias("_cy"),
            F.col("count"),
        )
        window_counts = shifted if window_counts is None else window_counts.unionAll(shifted)
    window_counts = window_counts.groupBy("_cx", "_cy").agg(
        F.sum("count").alias(label)
    )
    return (
        with_cell.join(window_counts, on=["_cx", "_cy"], how="left")
        .withColumn(label, F.col(label) - 1)  # exclude the shop itself
        .drop("_cx", "_cy")
    )


def phase1_features(spark, metrics, divisions=None):
    t0 = time.time()
    try:
        # warehouse built by partition_dataset.py — a divisions filter prunes to just those partition dirs
        shops = spark.read.parquet(f"{HDFS}/warehouse/shops")
        targets = spark.read.parquet(f"{HDFS}/warehouse/targets")
        if divisions:
            targets = targets.filter(F.col("division").isin(divisions))
        targets = targets.drop("division")
        source = "partitioned warehouse (partition pruning active)"
    except Exception:
        shops = spark.read.csv(f"{HDFS}/raw/shop_info_with_geo.csv", header=True, inferSchema=True)
        shops = shops.withColumn("shop_code", F.col("shop_code").cast("string"))
        targets = spark.read.csv(f"{HDFS}/raw/product_target_for_shop.csv", header=True, inferSchema=True)
        targets = targets.withColumn("shop_code", F.col("shop_code").cast("string"))
        source = "raw CSVs (run partition_dataset.py to enable pruning)"
        if divisions:
            shops = shops.filter(F.col("division").isin(divisions))
    if divisions and "warehouse" in source:
        shops = shops.filter(F.col("division").isin(divisions))

    n_raw = shops.count()
    metrics["run_scope"] = {
        "divisions": divisions or "all",
        "input_source": source,
    }
    shops = shops.dropDuplicates(["shop_code"]).filter(
        F.col("latitude").between(LAT_MIN, LAT_MAX)
        & F.col("longitude").between(LON_MIN, LON_MAX)
    )
    for col in ONEHOT_SOURCE + ["zila", "upazila"]:
        shops = shops.withColumn(col, F.coalesce(F.col(col), F.lit("Unknown")))

    for radius in DENSITY_RADII_KM:
        label = f"density_{int(radius * 1000)}m" if radius < 1 else f"density_{radius:g}km"
        shops = grid_density(shops, radius, label)

    centroids = shops.groupBy("zila").agg(
        F.avg("latitude").alias("_clat"), F.avg("longitude").alias("_clon")
    )
    shops = shops.join(centroids, "zila")
    # haversine distance to district centroid, in km
    shops = shops.withColumn(
        "dist_to_district_centre_km",
        2 * 6371.0 * F.asin(F.sqrt(
            F.pow(F.sin(F.radians(F.col("latitude") - F.col("_clat")) / 2), 2)
            + F.cos(F.radians("latitude")) * F.cos(F.radians("_clat"))
            * F.pow(F.sin(F.radians(F.col("longitude") - F.col("_clon")) / 2), 2)
        )),
    ).drop("_clat", "_clon")

    dedup = Window.partitionBy("shop_code").orderBy(F.col("Total").asc_nulls_last())
    targets = (
        targets.withColumn("_r", F.row_number().over(dedup)).filter("_r = 1").drop("_r")
    )
    n_products = sum(
        F.when(F.coalesce(F.col(c), F.lit(0.0)) > 0, 1).otherwise(0) for c in PRODUCT_COLS
    )
    targets = targets.select(
        "shop_code",
        F.col("Total").alias("total_target"),
        n_products.alias("n_products_targeted"),
    )
    features = (
        shops.join(targets, "shop_code", "left")
        .withColumn("has_target", F.col("total_target").isNotNull())
        .drop("geometry")
    )
    features.write.mode("overwrite").parquet(f"{HDFS}/processed/features.parquet")
    metrics["phase_1_spark"] = {
        "engine": "Spark SQL on HDFS",
        "n_raw_shop_rows": n_raw,
        "n_final_shops": features.count(),
        "n_shops_with_target": features.filter("has_target").count(),
        "density_method": "3x3 grid-window approximation (cell edge = radius)",
        "duration_sec": round(time.time() - t0, 1),
    }
    return features


def assemble(features):
    numeric = ["latitude", "longitude", "density_500m", "density_1km",
               "density_5km", "dist_to_district_centre_km", "n_products_targeted"]
    df = features.withColumn(
        "n_products_targeted", F.coalesce("n_products_targeted", F.lit(0))
    )
    onehot_cols = []
    for col in ONEHOT_SOURCE:
        # small cardinality: literal indicator columns let every phase reuse the frame without refitting indexers
        levels = [r[0] for r in df.select(col).distinct().collect()]
        for level in levels:
            name = f"{col}__{level}"
            df = df.withColumn(name, (F.col(col) == level).cast("double"))
            onehot_cols.append(name)
    assembler = VectorAssembler(inputCols=numeric + onehot_cols, outputCol="raw_features")
    df = assembler.transform(df)
    scaler = StandardScaler(inputCol="raw_features", outputCol="features",
                            withMean=True, withStd=True)
    return scaler.fit(df).transform(df).cache()


def phase2_clustering(df, metrics):
    t0 = time.time()
    evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette")
    scores, models = {}, {}
    for k in K_RANGE:
        model = KMeans(k=k, seed=SEED, featuresCol="features").fit(df)
        pred = model.transform(df)
        scores[k] = float(evaluator.evaluate(pred))
        models[k] = model
    best_k = max(scores, key=scores.get)
    kmeans_model = models[best_k]
    clustered = kmeans_model.transform(df).withColumnRenamed("prediction", "cluster")

    bisecting = BisectingKMeans(k=best_k, seed=SEED, featuresCol="features").fit(df)
    bisecting_sil = float(evaluator.evaluate(bisecting.transform(df)))

    clustered.select("shop_code", "cluster").write.mode("overwrite").parquet(
        f"{HDFS}/processed/clusters.parquet"
    )
    metrics["phase_2_spark"] = {
        "engine": "Spark MLlib KMeans + BisectingKMeans on HDFS",
        "silhouette_by_k": {str(k): round(v, 4) for k, v in scores.items()},
        "best_k": best_k,
        "bisecting_kmeans_silhouette": round(bisecting_sil, 4),
        "duration_sec": round(time.time() - t0, 1),
    }
    return clustered, kmeans_model


def phase3_regression(df, metrics):
    t0 = time.time()
    labelled = df.filter("has_target").withColumn("label", F.col("total_target"))
    train, test = labelled.randomSplit([0.8, 0.2], seed=SEED)
    models = {
        "linear_regression": LinearRegression(featuresCol="features"),
        "random_forest": RandomForestRegressor(
            featuresCol="features", numTrees=40, maxDepth=10, seed=SEED
        ),
        "gbt": GBTRegressor(featuresCol="features", maxIter=40, maxDepth=6, seed=SEED),
    }
    results = {}
    for name, algo in models.items():
        fitted = algo.fit(train)
        pred = fitted.transform(test)
        results[name] = {
            metric: float(
                RegressionEvaluator(metricName=metric).evaluate(pred)
            )
            for metric in ("rmse", "mae", "r2")
        }
    best = min(results, key=lambda n: results[n]["rmse"])
    metrics["phase_3_spark"] = {
        "engine": "Spark MLlib regression on HDFS (GBT ~ distributed XGBoost stand-in)",
        "models": results,
        "best_model": best,
        "duration_sec": round(time.time() - t0, 1),
    }


def phase4_classification(df, metrics):
    t0 = time.time()
    labelled = df.filter("has_target")
    peer = Window.partitionBy("cluster", "shop_type")
    labelled = labelled.withColumn(
        "peer_median", F.expr("percentile_approx(total_target, 0.5)").over(peer)
    ).withColumn(
        "label", (F.col("total_target") < RISK_THRESHOLD * F.col("peer_median")).cast("double")
    )
    train, test = labelled.randomSplit([0.8, 0.2], seed=SEED)
    # class weights instead of SMOTE (not available distributed in MLlib);
    # base rate comes from the training split only — the test rows never leak in
    base_rate = train.agg(F.avg("label")).first()[0]
    weigh = lambda df: df.withColumn(
        "weight",
        F.when(F.col("label") == 1.0, float(1 / max(base_rate, 1e-9)))
        .otherwise(float(1 / max(1 - base_rate, 1e-9))),
    )
    train, test = weigh(train), weigh(test)
    models = {
        "logistic_regression": LogisticRegression(featuresCol="features", weightCol="weight"),
        "random_forest": RandomForestClassifier(
            featuresCol="features", weightCol="weight", numTrees=40, maxDepth=10, seed=SEED
        ),
        "gbt": GBTClassifier(featuresCol="features", weightCol="weight",
                             maxIter=40, maxDepth=6, seed=SEED),
    }
    results = {}
    for name, algo in models.items():
        fitted = algo.fit(train)
        pred = fitted.transform(test)
        results[name] = {
            "roc_auc": float(BinaryClassificationEvaluator().evaluate(pred)),
            # weighted multiclass averages — NOT comparable with the sklearn lane's
            # positive-class f1/precision/recall, so the keys say so
            "f1_weighted": float(MulticlassClassificationEvaluator(metricName="f1").evaluate(pred)),
            "precision_weighted": float(
                MulticlassClassificationEvaluator(metricName="weightedPrecision").evaluate(pred)
            ),
            "recall_weighted": float(
                MulticlassClassificationEvaluator(metricName="weightedRecall").evaluate(pred)
            ),
        }
    best = max(results, key=lambda n: results[n]["roc_auc"])
    metrics["phase_4_spark"] = {
        "engine": "Spark MLlib classification on HDFS (class weights instead of SMOTE)",
        "label_base_rate": round(float(base_rate), 4),
        "models": results,
        "best_model": best,
        "duration_sec": round(time.time() - t0, 1),
    }


def phase5_anomaly(clustered, kmeans_model, metrics):
    t0 = time.time()
    centers = [(i, [float(x) for x in c]) for i, c in enumerate(kmeans_model.clusterCenters())]
    centers_df = clustered.sparkSession.createDataFrame(centers, ["cluster", "center"])
    scored = (
        clustered.withColumn("farr", vector_to_array("features"))
        .join(centers_df, "cluster")
        .withColumn(
            "centroid_dist",
            F.sqrt(F.aggregate(
                F.zip_with("farr", "center", lambda a, b: (a - b) * (a - b)),
                F.lit(0.0), lambda acc, x: acc + x,
            )),
        )
        .withColumn(
            "anomaly_rank",
            F.percent_rank().over(Window.partitionBy("cluster").orderBy("centroid_dist")),
        )
    )
    top = scored.filter("anomaly_rank >= 0.9")
    top.select(
        "shop_code", "cluster", "division", "zila", "shop_type",
        "centroid_dist", "anomaly_rank", "total_target",
    ).write.mode("overwrite").parquet(f"{HDFS}/processed/top_anomalies.parquet")
    metrics["phase_5_spark"] = {
        "engine": "Spark centroid-distance ranking on HDFS (distributed IForest/LOF stand-in)",
        "n_scored": scored.count(),
        "n_top_decile": top.count(),
        "duration_sec": round(time.time() - t0, 1),
    }


def main():
    # region scope from argv or DIVISIONS env, e.g. "Khulna,Barisal"
    raw_scope = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DIVISIONS", "")
    divisions = [d.strip() for d in raw_scope.split(",") if d.strip()] or None

    spark = (
        SparkSession.builder.appName(
            "smart-retail-mllib-pipeline"
            + (f"-{'-'.join(divisions)}" if divisions else "")
        )
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:8020")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    metrics = {"storage": "hdfs://namenode:8020/smart_retail"}

    features = phase1_features(spark, metrics, divisions)
    df = assemble(features)
    clustered, kmeans_model = phase2_clustering(df, metrics)
    phase3_regression(clustered, metrics)
    phase4_classification(clustered, metrics)
    phase5_anomaly(clustered, kmeans_model, metrics)

    metrics["total_duration_sec"] = round(
        sum(v["duration_sec"] for k, v in metrics.items() if isinstance(v, dict) and "duration_sec" in v), 1
    )
    # regional runs get their own metrics file; only a full run owns the default
    out_path = (
        LOCAL_METRICS.replace(".json", f"_{'_'.join(divisions)}.json")
        if divisions else LOCAL_METRICS
    )
    with open(out_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
