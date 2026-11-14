"""CarIntel Spark Processing Package."""
from config.spark_config import SparkConfig
SparkConfig.setup_environment()

from .schemas import RAW_CAR_SCHEMA, PROCESSED_CAR_SCHEMA

__all__ = ["RAW_CAR_SCHEMA", "PROCESSED_CAR_SCHEMA"]
