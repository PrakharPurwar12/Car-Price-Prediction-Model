"""
tests/test_ml_pipeline.py
Validates Spark MLlib feature transformation, pipeline serialization,
and inference robustness (specifically verifying the Peugeot edge-case fix).
"""

from pathlib import Path
import pytest
import config.spark_config
from config.spark_config import get_spark_session
from ml.feature_pipeline import build_feature_pipeline
from ml.predict import CarIntelPredictor, DEFAULT_RF_MODEL_PATH

ROOT_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def spark():
    session = get_spark_session("CarIntel-TestML")
    yield session
    session.stop()


def test_feature_pipeline_stages(spark):
    pipeline = build_feature_pipeline(include_scaler=False)
    stages = pipeline.getStages()
    assert len(stages) >= 3  # Indexers, Encoder, Assembler


def test_predictor_known_brand(spark):
    predictor = CarIntelPredictor(spark, DEFAULT_RF_MODEL_PATH)
    sample_car = {
        "brand": "Maruti",
        "year": 2017,
        "km_driven": 45000,
        "fuel": "Diesel",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner",
        "mileage_kmpl": 24.3,
        "engine_cc": 1248.0,
        "max_power_bhp": 74.0,
        "seats": 5
    }
    pred_price = predictor.predict(sample_car)
    assert pred_price > 50000.0
    assert pred_price < 5000000.0


def test_predictor_peugeot_edge_case_no_crash(spark):
    """
    CRITICAL AUDIT VERIFICATION:
    In the legacy app.py, brand 'Peugeot' crashed with KeyError: 19 because it
    was missing from the hardcoded label mapping.
    This test verifies that CarIntel's MLlib Pipeline handles 'Peugeot' gracefully.
    """
    predictor = CarIntelPredictor(spark, DEFAULT_RF_MODEL_PATH)
    peugeot_car = {
        "brand": "Peugeot",
        "year": 2018,
        "km_driven": 35000,
        "fuel": "Diesel",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner",
        "mileage_kmpl": 20.0,
        "engine_cc": 1498.0,
        "max_power_bhp": 95.0,
        "seats": 5
    }
    # Must execute without throwing KeyError or AnalysisException
    pred_price = predictor.predict(peugeot_car)
    assert isinstance(pred_price, float)
    assert pred_price > 15000.0
