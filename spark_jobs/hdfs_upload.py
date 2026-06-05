"""Upload the raw CSVs into HDFS via the Hadoop FileSystem API.

The apache/hadoop image runs under qemu emulation on Apple Silicon and its
`hdfs dfs -put` intermittently aborts mid-transfer; this uploader runs on the
native-arm64 Spark image instead (Spark bundles the Hadoop client), which
makes the 200 MB transfer reliable.

    spark-submit --master "local[1]" /opt/jobs/hdfs_upload.py
"""

from pyspark.sql import SparkSession

FILES = [
    "shop_info_with_geo.csv",
    "product_target_for_shop.csv",
    "product_info.csv",
]


def main() -> None:
    spark = (
        SparkSession.builder.appName("smart-retail-hdfs-upload")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:8020")
        .config("spark.hadoop.dfs.replication", "2")
        .getOrCreate()
    )
    jvm = spark._jvm
    conf = spark._jsc.hadoopConfiguration()
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(conf)
    Path = jvm.org.apache.hadoop.fs.Path

    fs.mkdirs(Path("/smart_retail/raw"))
    for name in FILES:
        src = Path(f"file:///data/dataset/{name}")
        dst = Path(f"/smart_retail/raw/{name}")
        fs.copyFromLocalFile(False, True, src, dst)  # keep local, overwrite dest
        status = fs.getFileStatus(dst)
        print(f"uploaded {name}: {status.getLen():,} bytes, "
              f"replication {status.getReplication()}")
    spark.stop()


if __name__ == "__main__":
    main()
