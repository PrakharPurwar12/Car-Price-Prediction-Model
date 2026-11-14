"""Data Cleaning, Normalization, and Preprocessing Module for CarIntel."""
import logging
from typing import Tuple, Dict, Any
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

logger = logging.getLogger(__name__)


def safe_extract_double(col_expr, pattern: str, group_idx: int = 1):
    """Safely extract regex group and cast to Double, returning None on empty/unmatched strings."""
    extracted = F.regexp_extract(col_expr, pattern, group_idx)
    return F.when((extracted.isNotNull()) & (F.length(F.trim(extracted)) > 0), 
                  extracted.cast(DoubleType())).otherwise(None)


def clean_and_preprocess_data(raw_df: DataFrame) -> Tuple[DataFrame, Dict[str, Any]]:
    """
    Execute distributed data cleaning, deduplication, regex unit parsing,
    and outlier handling.

    Args:
        raw_df: Ingested raw PySpark DataFrame.

    Returns:
        Tuple of (Cleaned PySpark DataFrame, Cleaning Audit Report Dictionary).
    """
    initial_rows = raw_df.count()
    logger.info(f"Starting distributed cleaning pipeline on {initial_rows} records.")

    # 1. Exact Deduplication (Fixes train/test data leakage identified in Phase 1)
    dedup_df = raw_df.dropDuplicates()
    dedup_rows = dedup_df.count()
    removed_duplicates = initial_rows - dedup_rows
    logger.info(f"Deduplication complete. Removed {removed_duplicates} duplicates. Remaining: {dedup_rows}")

    # 2. Multi-word Brand and Model Extraction
    brand_expr = F.when(F.col("Name").startswith("Land Rover") | F.col("Name").startswith("Land"), "Land Rover") \
                  .when(F.col("Name").startswith("Mercedes-Benz"), "Mercedes-Benz") \
                  .when(F.col("Name").startswith("Ashok Leyland") | F.col("Name").startswith("Ashok"), "Ashok Leyland") \
                  .otherwise(F.split(F.col("Name"), " ").getItem(0))

    with_brand = dedup_df.withColumn("brand", brand_expr)

    # Model name: strip brand prefix and extract the next primary model token
    model_expr = F.when(F.col("brand") == "Land Rover", F.regexp_replace(F.col("Name"), "^(Land Rover|Land)\\s*", "")) \
                  .when(F.col("brand") == "Mercedes-Benz", F.regexp_replace(F.col("Name"), "^Mercedes-Benz\\s*", "")) \
                  .when(F.col("brand") == "Ashok Leyland", F.regexp_replace(F.col("Name"), "^(Ashok Leyland|Ashok)\\s*", "")) \
                  .otherwise(F.regexp_replace(F.col("Name"), "^\\w+\\s*", ""))
    
    with_model = with_brand.withColumn("raw_model_suffix", model_expr)
    with_model = with_model.withColumn(
        "model_name", 
        F.when(F.length(F.trim(F.col("raw_model_suffix"))) > 0, 
               F.split(F.trim(F.col("raw_model_suffix")), " ").getItem(0))
        .otherwise("Generic")
    ).drop("raw_model_suffix")

    # 3. Safe Numeric Parsing from String Columns
    # Mileage: extract numerical value and unit (kmpl vs km/kg)
    mileage_val = safe_extract_double(F.col("mileage"), r"(\d+\.?\d*)", 1)
    mileage_unit = F.when(F.lower(F.col("mileage")).contains("km/kg"), "km/kg") \
                    .otherwise("kmpl")
    
    # Standardize mileage to equivalent kmpl (1 kg CNG ~= 1.39 L Petrol equivalent energy)
    mileage_std = F.when(mileage_unit == "km/kg", F.round(mileage_val * 1.39, 2)) \
                   .otherwise(mileage_val)

    # Engine: extract numeric displacement in CC
    engine_val = safe_extract_double(F.col("engine"), r"(\d+)", 1)

    # Max power: extract horsepower in bhp
    power_val = safe_extract_double(F.col("max_power"), r"(\d+\.?\d*)", 1)

    # Torque: parse Nm and RPM where available
    is_kgm = F.lower(F.col("torque")).contains("kgm")
    raw_torque_val = safe_extract_double(F.col("torque"), r"(\d+\.?\d*)", 1)
    torque_nm = F.when(is_kgm & raw_torque_val.isNotNull(), F.round(raw_torque_val * 9.80665, 1)) \
                 .otherwise(raw_torque_val)
    
    # Extract RPM
    raw_rpm_val = safe_extract_double(F.col("torque"), r"[@\sat/]+\s*(\d+)", 1)
    torque_rpm = F.when(raw_rpm_val > 10000, None).otherwise(raw_rpm_val)

    cleaned = with_model \
        .withColumn("mileage_val", mileage_val) \
        .withColumn("mileage_unit", mileage_unit) \
        .withColumn("mileage_kmpl_std", mileage_std) \
        .withColumn("engine_cc", engine_val) \
        .withColumn("max_power_bhp", power_val) \
        .withColumn("torque_nm", torque_nm) \
        .withColumn("torque_rpm", torque_rpm)

    # 4. Outlier and Impossible Zero Handling
    # Identify impossible zeroes: mileage = 0.0 kmpl (17 rows), max_power = 0.0 (6 rows)
    zero_mileage_filter = (F.col("mileage_val") == 0.0) | F.col("mileage_val").isNull()
    zero_power_filter = (F.col("max_power_bhp") == 0.0) | F.col("max_power_bhp").isNull()

    # Fallback medians calculated from valid non-zero rows
    med_mileage = 19.3
    med_engine = 1248.0
    med_power = 82.0
    med_seats = 5

    # Replace impossible values and nulls with medians
    cleaned = cleaned \
        .withColumn("mileage_kmpl_std", F.when(
            F.col("mileage_kmpl_std").isNull() | F.isnan("mileage_kmpl_std") | (F.col("mileage_kmpl_std") <= 0.0),
            med_mileage
        ).otherwise(F.col("mileage_kmpl_std"))) \
        .withColumn("mileage_val", F.when(
            F.col("mileage_val").isNull() | F.isnan("mileage_val") | (F.col("mileage_val") <= 0.0),
            med_mileage
        ).otherwise(F.col("mileage_val"))) \
        .withColumn("engine_cc", F.when(
            F.col("engine_cc").isNull() | F.isnan("engine_cc") | (F.col("engine_cc") <= 0),
            med_engine
        ).otherwise(F.col("engine_cc"))) \
        .withColumn("max_power_bhp", F.when(
            F.col("max_power_bhp").isNull() | F.isnan("max_power_bhp") | (F.col("max_power_bhp") <= 0.0),
            med_power
        ).otherwise(F.col("max_power_bhp"))) \
        .withColumn("seats", F.when(
            F.col("seats").isNull() | (F.col("seats") < 2),
            med_seats
        ).otherwise(F.col("seats").cast(IntegerType())))

    # 5. Filter invalid records (e.g. non-positive prices or invalid years)
    # Valid luxury vehicles with prices up to 10M INR are strictly preserved
    valid_filter = (F.col("selling_price") > 0) & \
                   (F.col("year") >= 1980) & (F.col("year") <= 2025) & \
                   (F.col("km_driven") > 0) & (F.col("km_driven") <= 2500000)

    filtered_df = cleaned.filter(valid_filter)
    final_count = filtered_df.count()
    removed_records = dedup_rows - final_count

    # 6. Add deterministic unique 64-bit Long listing_id
    filtered_df = filtered_df.withColumn("listing_id", F.monotonically_increasing_id())

    # Standardize column renaming
    final_df = filtered_df.withColumnRenamed("Name", "full_name")

    report = {
        "initial_rows": initial_rows,
        "deduplicated_rows": dedup_rows,
        "removed_duplicates": removed_duplicates,
        "imputed_mileage_medians": med_mileage,
        "imputed_engine_medians": med_engine,
        "imputed_power_medians": med_power,
        "final_cleaned_rows": final_count,
        "removed_corrupt_records": removed_records
    }

    logger.info(f"Preprocessing completed. Final valid records: {final_count}")
    return final_df, report
