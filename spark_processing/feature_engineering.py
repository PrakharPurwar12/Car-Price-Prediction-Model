"""Feature Engineering Module for CarIntel Platform using PySpark."""
import logging
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
from .schemas import REFERENCE_YEAR

logger = logging.getLogger(__name__)

LUXURY_BRANDS = [
    "BMW", "Mercedes-Benz", "Audi", "Jaguar", "Land Rover", 
    "Volvo", "Lexus", "Jeep", "Porsche"
]


def engineer_features(df: DataFrame, reference_year: int = REFERENCE_YEAR) -> DataFrame:
    """
    Engineer domain-specific automotive features for analytical queries,
    GraphX graph construction, and MLlib prediction pipelines.

    Args:
        df: Cleaned PySpark DataFrame from preprocessing module.
        reference_year: Baseline year for vehicle age calculation (default: 2025).

    Returns:
        DataFrame enriched with engineered features.
    """
    logger.info("Executing feature engineering pipeline...")

    # 1. Vehicle Age (years)
    with_age = df.withColumn(
        "vehicle_age", 
        F.greatest(F.lit(0), F.lit(reference_year) - F.col("year")).cast(IntegerType())
    )

    # 2. Annual Odometer Usage (km per year)
    with_km_rate = with_age.withColumn(
        "km_per_year",
        F.round(F.col("km_driven") / (F.col("vehicle_age") + 0.5), 2).cast(DoubleType())
    )

    # 3. Log-Transformed Selling Price (Stabilizes skewed regression target)
    with_log_price = with_km_rate.withColumn(
        "log_selling_price",
        F.round(F.log1p(F.col("selling_price").cast(DoubleType())), 4)
    )

    # 4. Engine Power-to-Displacement Efficiency Ratio (bhp / CC)
    with_power_ratio = with_log_price.withColumn(
        "power_to_engine_ratio",
        F.round(F.col("max_power_bhp") / (F.col("engine_cc") + 1e-4), 4)
    )

    # 5. Market Segment Tier Indicator
    is_luxury = F.when(F.col("brand").isin(LUXURY_BRANDS), F.lit(1)).otherwise(F.lit(0))
    enriched_df = with_power_ratio.withColumn("is_luxury_brand", is_luxury)

    logger.info(f"Feature engineering completed. Output columns: {len(enriched_df.columns)}")
    return enriched_df

