"""Distributed ETL over the retail dataset on the Spark standalone cluster (run via spark-submit, see Makefile)."""

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

DATASET_DIR = "/data/dataset"
OUT_DIR = "/data/out/spark"

# Bangladesh bounding box (must match smart_retail.data.schema.BD_BBOX)
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 20.5, 26.7, 88.0, 92.7
PRODUCT_COLS = [f"prod_{i}" for i in range(1, 18)]


def main() -> None:
    spark = (
        SparkSession.builder.appName("smart-retail-etl")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    shops = spark.read.csv(
        f"{DATASET_DIR}/shop_info_with_geo.csv", header=True, inferSchema=True
    ).withColumn("shop_code", F.col("shop_code").cast("string"))

    targets = spark.read.csv(
        f"{DATASET_DIR}/product_target_for_shop.csv", header=True, inferSchema=True
    ).withColumn("shop_code", F.col("shop_code").cast("string"))

    n_raw = shops.count()

    # same cleaning rules as smart_retail.features.engineering, expressed as distributed Spark SQL
    shops = shops.dropDuplicates(["shop_code"])
    n_unique = shops.count()

    shops = shops.filter(
        F.col("latitude").between(LAT_MIN, LAT_MAX)
        & F.col("longitude").between(LON_MIN, LON_MAX)
    )
    n_valid_gps = shops.count()

    for col in ("organizer", "place_nature", "freq_type", "shop_type",
                "shop_style", "shop_importance", "ss_importance",
                "zila", "upazila", "division"):
        shops = shops.withColumn(col, F.coalesce(F.col(col), F.lit("Unknown")))

    # prefer valued target rows over all-NaN duplicates (2.4% of codes repeat)
    dedup_window = Window.partitionBy("shop_code").orderBy(
        F.col("Total").asc_nulls_last()
    )
    targets = (
        targets.withColumn("_rank", F.row_number().over(dedup_window))
        .filter(F.col("_rank") == 1)
        .drop("_rank")
    )
    n_products = sum(
        F.when(F.coalesce(F.col(c), F.lit(0.0)) > 0, 1).otherwise(0)
        for c in PRODUCT_COLS
    )
    targets = targets.select(
        "shop_code",
        F.col("Total").alias("total_target"),
        n_products.alias("n_products_targeted"),
    )

    clean = (
        shops.join(targets, on="shop_code", how="left")
        .withColumn("has_target", F.col("total_target").isNotNull())
        .drop("geometry")
    )
    n_with_target = clean.filter("has_target").count()

    division_stats = (
        clean.groupBy("division")
        .agg(
            F.count("*").alias("shops"),
            F.sum(F.col("has_target").cast("int")).alias("shops_with_target"),
            F.round(F.avg("total_target"), 2).alias("mean_target"),
            F.round(F.expr("percentile_approx(total_target, 0.5)"), 2).alias("median_target"),
            F.countDistinct("zila").alias("districts"),
            F.countDistinct("upazila").alias("upazilas"),
        )
        .orderBy(F.desc("shops"))
    )

    clean.coalesce(4).write.mode("overwrite").parquet(f"{OUT_DIR}/clean_shops.parquet")
    division_stats.coalesce(1).write.mode("overwrite").option("header", True).csv(
        f"{OUT_DIR}/division_stats.csv"
    )

    print("=" * 62)
    print("Spark distributed ETL — smart-retail")
    print(f"  raw shop rows          : {n_raw:,}")
    print(f"  unique shop codes      : {n_unique:,}")
    print(f"  valid GPS (inside BD)  : {n_valid_gps:,}")
    print(f"  joined with a target   : {n_with_target:,}")
    print(f"  output                 : {OUT_DIR}/clean_shops.parquet")
    print("=" * 62)
    division_stats.show(10, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
