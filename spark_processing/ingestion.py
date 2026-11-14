"""Data Ingestion Module for CarIntel using Apache Spark."""
import os
import logging
from typing import Tuple, Dict, Any
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, FloatType

from .schemas import RAW_CAR_SCHEMA
from config.spark_config import get_project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def ingest_raw_data(
    spark: SparkSession, 
    csv_path: str = None
) -> Tuple[DataFrame, Dict[str, Any]]:
    """
    Ingest raw CarDekho CSV using an explicit StructType schema.
    
    Args:
        spark: Active SparkSession.
        csv_path: Path to raw CSV. If None, uses default project-relative path.
        
    Returns:
        Tuple of (Raw PySpark DataFrame, Ingestion Audit Metrics Dictionary).
    """
    if csv_path is None:
        csv_path = str(get_project_root() / "data" / "raw" / "Car dekho - Car dekho.csv")
        if not os.path.exists(csv_path):
            csv_path = str(get_project_root() / "Car dekho - Car dekho.csv")

    logger.info(f"Ingesting raw dataset from: {csv_path}")

    # Explicit schema ingestion - never depend on inferSchema
    raw_df = spark.read \
        .option("header", "true") \
        .option("mode", "DROPMALFORMED") \
        .schema(RAW_CAR_SCHEMA) \
        .csv(csv_path)

    initial_row_count = raw_df.count()
    initial_col_count = len(raw_df.columns)
    logger.info(f"Raw dataset loaded: {initial_row_count} rows, {initial_col_count} columns")

    # Compute null counts across all columns safely by type
    null_counts = {}
    schema_map = {field.name: field.dataType for field in raw_df.schema.fields}
    for col_name in raw_df.columns:
        dt = schema_map.get(col_name)
        if isinstance(dt, (DoubleType, FloatType)):
            cond = F.col(col_name).isNull() | F.isnan(F.col(col_name))
        else:
            cond = F.col(col_name).isNull()
        cnt = raw_df.filter(cond).count()
        null_counts[col_name] = cnt

    # Compute duplicate count
    unique_count = raw_df.dropDuplicates().count()
    duplicate_count = initial_row_count - unique_count
    logger.info(f"Duplicate records identified: {duplicate_count} ({duplicate_count/initial_row_count*100:.2f}%)")

    metrics = {
        "source_path": csv_path,
        "initial_rows": initial_row_count,
        "initial_columns": initial_col_count,
        "duplicate_rows": duplicate_count,
        "null_counts": null_counts,
    }

    return raw_df, metrics
