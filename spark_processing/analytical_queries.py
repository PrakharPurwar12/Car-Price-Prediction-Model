"""
spark_processing/analytical_queries.py
Executes comprehensive Spark SQL analytical queries on processed vehicle data,
computes market intelligence metrics, and exports aggregated results as JSON/Parquet
for consumption by the Streamlit dashboard.
"""

import json
import logging
from pathlib import Path
from pyspark.sql import SparkSession
from config.spark_config import get_spark_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SparkSQLAnalytics")

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = str(ROOT_DIR / "data" / "processed" / "vehicles.parquet")
OUTPUT_ANALYTICS_DIR = ROOT_DIR / "data" / "processed" / "sql_analytics"


def run_analytical_queries(spark: SparkSession, parquet_path: str = PROCESSED_DATA_PATH):
    logger.info(f"Loading processed data from {parquet_path}")
    df = spark.read.parquet(parquet_path)
    df.createOrReplaceTempView("vehicles")
    OUTPUT_ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

    analytics_results = {}

    # 1. Brand Intelligence: Volume, Valuation, Mean, Median, Min, Max Price, Luxury Flag
    logger.info("Executing Query 1: Brand Market Intelligence...")
    q1 = """
    SELECT 
        brand,
        COUNT(*) AS vehicle_count,
        ROUND(AVG(selling_price), 2) AS avg_price,
        ROUND(PERCENTILE_APPROX(selling_price, 0.5), 2) AS median_price,
        ROUND(MIN(selling_price), 2) AS min_price,
        ROUND(MAX(selling_price), 2) AS max_price,
        ROUND(AVG(vehicle_age), 1) AS avg_age,
        ROUND(AVG(km_driven), 0) AS avg_km_driven,
        MAX(is_luxury_brand) AS is_luxury
    FROM vehicles
    GROUP BY brand
    ORDER BY vehicle_count DESC
    """
    df_brand = spark.sql(q1)
    analytics_results["brand_summary"] = [row.asDict() for row in df_brand.collect()]

    # 2. Fuel Type Depreciation Analysis
    logger.info("Executing Query 2: Fuel Type Depreciation Analysis...")
    q2 = """
    SELECT 
        fuel,
        COUNT(*) AS total_count,
        ROUND(AVG(selling_price), 2) AS avg_price,
        ROUND(AVG(mileage_kmpl_std), 2) AS avg_mileage,
        ROUND(AVG(km_driven), 0) AS avg_km_driven,
        ROUND(AVG(vehicle_age), 1) AS avg_age,
        ROUND(AVG(selling_price / (vehicle_age + 1)), 2) AS price_per_year_retained
    FROM vehicles
    GROUP BY fuel
    ORDER BY total_count DESC
    """
    df_fuel = spark.sql(q2)
    analytics_results["fuel_summary"] = [row.asDict() for row in df_fuel.collect()]

    # 3. Transmission Premium Analysis
    logger.info("Executing Query 3: Transmission Premium Analysis...")
    q3 = """
    SELECT 
        transmission,
        COUNT(*) AS total_count,
        ROUND(AVG(selling_price), 2) AS avg_price,
        ROUND(PERCENTILE_APPROX(selling_price, 0.5), 2) AS median_price,
        ROUND(AVG(max_power_bhp), 1) AS avg_power_bhp,
        ROUND(AVG(engine_cc), 0) AS avg_engine_cc
    FROM vehicles
    GROUP BY transmission
    """
    df_trans = spark.sql(q3)
    analytics_results["transmission_summary"] = [row.asDict() for row in df_trans.collect()]

    # 4. Owner Sequence Impact on Resale Value
    logger.info("Executing Query 4: Owner Impact on Resale Value...")
    q4 = """
    SELECT 
        owner,
        COUNT(*) AS count,
        ROUND(AVG(selling_price), 2) AS avg_price,
        ROUND(AVG(vehicle_age), 1) AS avg_age,
        ROUND(AVG(km_driven), 0) AS avg_km_driven
    FROM vehicles
    GROUP BY owner
    ORDER BY avg_price DESC
    """
    df_owner = spark.sql(q4)
    analytics_results["owner_summary"] = [row.asDict() for row in df_owner.collect()]

    # 5. Engine Displacement Segmentation
    logger.info("Executing Query 5: Engine Displacement Segmentation...")
    q5 = """
    SELECT 
        CASE 
            WHEN engine_cc < 1000 THEN 'Under 1.0L (<1000cc)'
            WHEN engine_cc >= 1000 AND engine_cc < 1500 THEN '1.0L - 1.5L (1000-1499cc)'
            WHEN engine_cc >= 1500 AND engine_cc < 2000 THEN '1.5L - 2.0L (1500-1999cc)'
            WHEN engine_cc >= 2000 AND engine_cc < 3000 THEN '2.0L - 3.0L (2000-2999cc)'
            ELSE '3.0L+ High Capacity (>=3000cc)'
        END AS engine_segment,
        COUNT(*) AS count,
        ROUND(AVG(selling_price), 2) AS avg_price,
        ROUND(AVG(max_power_bhp), 1) AS avg_power_bhp,
        ROUND(AVG(mileage_kmpl_std), 2) AS avg_mileage
    FROM vehicles
    GROUP BY 1
    ORDER BY avg_price ASC
    """
    df_engine = spark.sql(q5)
    analytics_results["engine_segment_summary"] = [row.asDict() for row in df_engine.collect()]

    # 6. Overall Market KPI Overview
    logger.info("Executing Query 6: Overall Market KPI Overview...")
    q6 = """
    SELECT 
        COUNT(*) AS total_vehicles,
        COUNT(DISTINCT brand) AS total_brands,
        COUNT(DISTINCT model_name) AS total_models,
        ROUND(AVG(selling_price), 2) AS market_avg_price,
        ROUND(PERCENTILE_APPROX(selling_price, 0.5), 2) AS market_median_price,
        ROUND(AVG(vehicle_age), 1) AS market_avg_age,
        ROUND(AVG(km_driven), 0) AS market_avg_km_driven,
        ROUND(SUM(selling_price) / 10000000.0, 2) AS total_market_value_crores
    FROM vehicles
    """
    df_kpi = spark.sql(q6)
    analytics_results["market_kpi"] = df_kpi.collect()[0].asDict()

    # Helper to convert decimals/floats
    def json_serializer(obj):
        import decimal
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    # Save to JSON
    json_path = OUTPUT_ANALYTICS_DIR / "market_analytics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analytics_results, f, indent=2, default=json_serializer)
    logger.info(f"Market analytics saved successfully to: {json_path}")

    return analytics_results


if __name__ == "__main__":
    spark = get_spark_session("CarIntel-AnalyticsQueries")
    try:
        run_analytical_queries(spark)
    finally:
        spark.stop()
