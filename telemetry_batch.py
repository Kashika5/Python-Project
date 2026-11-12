"""
Graded Assignment - Big Data Platforms & Analytics
Resilient Global Telemetry Platform: PySpark batch processing

Covers:
  3.1 Transformations / actions and narrow vs wide dependencies
  3.2 Data-skew mitigation with salting + hash partitioning
  3.4 Checkpointing to truncate a long lineage in an iterative job

Run:  spark-submit telemetry_batch.py
(With no input path available, a synthetic skewed dataset is generated so the
script runs end to end. Replace INPUT_PATH with the real Parquet location.)
"""
import os
import random
import tempfile
import sys
if sys.platform.startswith("win"):
    os.environ["HADOOP_HOME"] = r"C:\hadoop"
    os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ["PATH"]

# Make Spark's workers use the same Python as this script
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
from pyspark.sql import SparkSession, functions as F

INPUT_PATH = None                      # e.g. "hdfs:///telemetry/historical/"
CHECKPOINT_DIR = os.path.join(tempfile.gettempdir(), "telemetry_ckpt")  # use hdfs:/// in production
SALT_BUCKETS = 16
CHECKPOINT_EVERY = 25
ITERATIONS = 300

spark = (SparkSession.builder
         .appName("TelemetryBatch")
         .config("spark.python.worker.faulthandler.enabled", "true")
         .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
         .config("spark.sql.shuffle.partitions", "50")
         .config("spark.sql.adaptive.enabled", "true")
         .config("spark.sql.adaptive.skewJoin.enabled", "true")
         .getOrCreate())
sc = spark.sparkContext
sc.setLogLevel("WARN")
sc.setCheckpointDir(CHECKPOINT_DIR)


# ----------------------------------------------------------------------------
# Ingest (synthetic data if no real path): truck "T0" is ~1000x heavier
# ----------------------------------------------------------------------------
def synthetic_rows():
    random.seed(7)
    models = ["Volta-X", "Haul-Pro", "City-Van"]
    rows = []
    for v in range(30):
        n = 20000 if v == 0 else 20            # data skew: vehicle 0 = 1000x logs
        model = models[v % 3]
        for _ in range(n):
            rows.append((f"T{v}", model, random.uniform(60, 120)))
    return rows


if INPUT_PATH:
    raw = spark.read.parquet(INPUT_PATH)                                   # lazy
else:
    raw = spark.createDataFrame(synthetic_rows(),
                                ["vehicle_id", "vehicle_model", "engine_temp"])

# ----------------------------------------------------------------------------
# 3.1 Transformations and actions
# ----------------------------------------------------------------------------
clean = (raw
         .select("vehicle_id", "vehicle_model", "engine_temp")             # NARROW
         .filter(F.col("engine_temp").isNotNull() &
                 F.col("engine_temp").between(-40, 200)))                  # NARROW

avg_temp = (clean.groupBy("vehicle_model")                                 # WIDE (shuffle)
                 .agg(F.avg("engine_temp").alias("avg_engine_temp")))

print("=== Naive average engine temp per model ===")
avg_temp.orderBy("vehicle_model").show()                                   # ACTION

# ----------------------------------------------------------------------------
# 3.2 Salting to mitigate skew (average = total_sum / total_count)
# ----------------------------------------------------------------------------
salted = clean.withColumn("salt", (F.rand(seed=1) * SALT_BUCKETS).cast("int"))   # NARROW

partial = (salted
           .repartition(50, "vehicle_id", "salt")        # HASH partitioning on salted key (WIDE)
           .groupBy("vehicle_model", "vehicle_id", "salt")                 # WIDE, hot key split in 16
           .agg(F.sum("engine_temp").alias("temp_sum"),
                F.count("engine_temp").alias("temp_cnt")))

per_vehicle = (partial
               .groupBy("vehicle_model", "vehicle_id")                     # WIDE, small data now
               .agg(F.sum("temp_sum").alias("temp_sum"),
                    F.sum("temp_cnt").alias("temp_cnt")))

model_avg = (per_vehicle
             .groupBy("vehicle_model")                                     # WIDE
             .agg((F.sum("temp_sum") / F.sum("temp_cnt")).alias("avg_engine_temp")))

print("=== Salted average engine temp per model (must match the naive result) ===")
model_avg.orderBy("vehicle_model").show()                                  # ACTION

# Range partitioning alternative (only for ordered output / range queries):
# clean.repartitionByRange(50, "vehicle_id")

# Inspect the skew before/after: rows per partition after hash on vehicle_id only
def part_sizes(df):
    return df.rdd.glom().map(len).collect()

print("Max partition size, hash(vehicle_id) only :",
      max(part_sizes(clean.repartition(8, "vehicle_id"))))
print("Max partition size, hash(vehicle_id,salt) :",
      max(part_sizes(salted.repartition(8, "vehicle_id", "salt"))))

# ----------------------------------------------------------------------------
# 3.4 Iterative job + checkpointing to truncate lineage
# ----------------------------------------------------------------------------
state = (clean.select("vehicle_id", "engine_temp").rdd
         .map(lambda r: (r["vehicle_id"], float(r["engine_temp"])))
         .reduceByKey(lambda a, b: (a + b) / 2.0))

def lineage_depth(rdd):
    return len(rdd.toDebugString().decode("utf-8").splitlines())

for i in range(1, ITERATIONS + 1):
    state = state.mapValues(lambda v: 0.9 * v + 0.1 * 90.0)   # each loop deepens lineage

    if i % CHECKPOINT_EVERY == 0:
        before = lineage_depth(state)
        state.cache()          # keep in memory so the checkpoint write doesn't recompute
        state.checkpoint()     # mark for checkpoint (lazy)
        state.count()          # ACTION: materialises the checkpoint -> DAG truncated
        print(f"iter {i:3d}: lineage lines before={before:4d}  after={lineage_depth(state):3d}"
              f"  checkpointed={state.isCheckpointed()}")

print("Sample final state:", state.take(3))
spark.stop()
