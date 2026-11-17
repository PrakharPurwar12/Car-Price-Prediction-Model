"""Explicit PySpark StructType Schemas for CarIntel Data Ingestion and ETL."""
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    LongType,
    DoubleType,
)

# Explicit schema for reading raw CSV (no inferSchema)
RAW_CAR_SCHEMA = StructType([
    StructField("Name", StringType(), False),
    StructField("year", IntegerType(), False),
    StructField("selling_price", LongType(), False),
    StructField("km_driven", IntegerType(), False),
    StructField("fuel", StringType(), False),
    StructField("seller_type", StringType(), False),
    StructField("transmission", StringType(), False),
    StructField("owner", StringType(), False),
    StructField("mileage", StringType(), True),
    StructField("engine", StringType(), True),
    StructField("max_power", StringType(), True),
    StructField("torque", StringType(), True),
    StructField("seats", DoubleType(), True),
])

# Explicit schema for processed clean vehicle dataset
PROCESSED_CAR_SCHEMA = StructType([
    StructField("listing_id", LongType(), False),
    StructField("full_name", StringType(), False),
    StructField("brand", StringType(), False),
    StructField("model_name", StringType(), False),
    StructField("year", IntegerType(), False),
    StructField("vehicle_age", IntegerType(), False),
    StructField("selling_price", LongType(), False),
    StructField("log_selling_price", DoubleType(), False),
    StructField("km_driven", IntegerType(), False),
    StructField("km_per_year", DoubleType(), False),
    StructField("fuel", StringType(), False),
    StructField("seller_type", StringType(), False),
    StructField("transmission", StringType(), False),
    StructField("owner", StringType(), False),
    StructField("mileage_val", DoubleType(), False),
    StructField("mileage_unit", StringType(), False),
    StructField("mileage_kmpl_std", DoubleType(), False),
    StructField("engine_cc", DoubleType(), False),
    StructField("max_power_bhp", DoubleType(), False),
    StructField("torque_nm", DoubleType(), True),
    StructField("torque_rpm", DoubleType(), True),
    StructField("seats", IntegerType(), False),
    StructField("power_to_engine_ratio", DoubleType(), False),
    StructField("is_luxury_brand", IntegerType(), False),
])

# Baseline reference year for vehicle age calculation across ETL, ML and inference
REFERENCE_YEAR = 2025

