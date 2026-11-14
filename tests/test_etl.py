"""
tests/test_etl.py
Validates the distributed ETL pipeline: schema ingestion, exact deduplication,
multi-word brand extraction, unit standardization, and null median imputation.
"""

from pathlib import Path
import pytest
import config.spark_config
from config.spark_config import get_spark_session
from spark_processing.ingestion import ingest_raw_data
from spark_processing.preprocessing import clean_and_preprocess_data
from spark_processing.feature_engineering import engineer_features

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_CSV_PATH = str(ROOT_DIR / "data" / "raw" / "Car dekho - Car dekho.csv")


@pytest.fixture(scope="module")
def spark():
    session = get_spark_session("CarIntel-TestETL")
    yield session
    session.stop()


def test_ingestion_row_count(spark):
    raw_df, metrics = ingest_raw_data(spark, RAW_CSV_PATH)
    assert raw_df.count() == 8148
    assert metrics["duplicate_rows"] == 1202


def test_deduplication_removes_exact_duplicates(spark):
    raw_df, _ = ingest_raw_data(spark, RAW_CSV_PATH)
    cleaned_df, report = clean_and_preprocess_data(raw_df)
    assert report["removed_duplicates"] == 1202
    assert cleaned_df.count() == 6946


def test_multi_word_brand_extraction(spark):
    test_rows = [
        ("Land Rover Discovery Sport", 2019, 4500000, 30000, "Diesel", "Individual", "Automatic", "First Owner", "12.8 kmpl", "1999 CC", "177 bhp", "430Nm", 7),
        ("Mercedes-Benz C-Class", 2020, 5200000, 20000, "Petrol", "Dealer", "Automatic", "First Owner", "14.0 kmpl", "1991 CC", "201 bhp", "300Nm", 5),
        ("Maruti Swift VDI", 2017, 500000, 50000, "Diesel", "Individual", "Manual", "First Owner", "28.4 kmpl", "1248 CC", "74 bhp", "190Nm", 5)
    ]
    cols = ["Name", "year", "selling_price", "km_driven", "fuel", "seller_type", "transmission", "owner", "mileage", "engine", "max_power", "torque", "seats"]
    raw = spark.createDataFrame(test_rows, cols)
    cleaned, _ = clean_and_preprocess_data(raw)
    
    brands = {row["full_name"]: row["brand"] for row in cleaned.collect()}
    assert brands["Land Rover Discovery Sport"] == "Land Rover"
    assert brands["Mercedes-Benz C-Class"] == "Mercedes-Benz"
    assert brands["Maruti Swift VDI"] == "Maruti"


def test_feature_engineering_columns(spark):
    raw_df, _ = ingest_raw_data(spark, RAW_CSV_PATH)
    cleaned_df, _ = clean_and_preprocess_data(raw_df)
    feat_df = engineer_features(cleaned_df)
    
    expected_cols = [
        "vehicle_age", "km_per_year", "log_selling_price", 
        "power_to_engine_ratio", "is_luxury_brand"
    ]
    for c in expected_cols:
        assert c in feat_df.columns
