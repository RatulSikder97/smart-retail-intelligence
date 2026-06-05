"""Build the division-partitioned warehouse in HDFS.

Raw CSVs are monolithic; this job rewrites them as parquet partitioned by
division (Hive layout: .../shops/division=Dhaka/part-*.parquet). Spark's
partition pruning then lets any downstream job touch only the divisions it
was asked to process — the semantically-regional distribution the raw files
cannot offer. Targets carry no division column, so it is attached via the
shop join before partitioning.

    spark-submit --master spark://spark-master:7077 /opt/jobs/partition_dataset.py
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

HDFS = "hdfs://namenode:8020/smart_retail"
WAREHOUSE = f"{HDFS}/warehouse"


def main() -> None:
    spark = (
        SparkSession.builder.appName("smart-retail-partition-warehouse")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:8020")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    shops = (
        spark.read.csv(f"{HDFS}/raw/shop_info_with_geo.csv", header=True, inferSchema=True)
        .withColumn("shop_code", F.col("shop_code").cast("string"))
        .withColumn("division", F.coalesce("division", F.lit("Unknown")))
    )
    targets = (
        spark.read.csv(f"{HDFS}/raw/product_target_for_shop.csv", header=True, inferSchema=True)
        .withColumn("shop_code", F.col("shop_code").cast("string"))
    )

    (
        shops.repartition("division")
        .write.mode("overwrite")
        .partitionBy("division")
        .parquet(f"{WAREHOUSE}/shops")
    )

    # attach division to targets through the shop join, then partition the same way
    division_of = shops.select("shop_code", "division").dropDuplicates(["shop_code"])
    (
        targets.join(division_of, "shop_code", "left")
        .withColumn("division", F.coalesce("division", F.lit("Unknown")))
        .repartition("division")
        .write.mode("overwrite")
        .partitionBy("division")
        .parquet(f"{WAREHOUSE}/targets")
    )

    per_division = (
        spark.read.parquet(f"{WAREHOUSE}/shops")
        .groupBy("division").count().orderBy(F.desc("count"))
    )
    print("=" * 56)
    print("Division-partitioned warehouse written to", WAREHOUSE)
    per_division.show(12, truncate=False)
    spark.stop()


if __name__ == "__main__":
    main()
