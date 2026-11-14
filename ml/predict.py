"""
ml/predict.py
Production inference engine for CarIntel:
Loads saved Spark MLlib PipelineModel and computes vehicle price predictions
given vehicle input features.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Union

import config.spark_config
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from config.spark_config import get_spark_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CarIntelPredictor")

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RF_MODEL_PATH = str(ROOT_DIR / "ml" / "models" / "rf_model")
DEFAULT_GBT_MODEL_PATH = str(ROOT_DIR / "ml" / "models" / "gbt_model")

# Luxury brand lookup
LUXURY_BRANDS = {
    "audi", "bmw", "mercedes-benz", "jaguar", "land rover",
    "volvo", "lexus", "porsche", "bentley", "ferrari",
    "rolls-royce", "maserati", "lamborghini"
}


class CarIntelPredictor:
    def __init__(self, spark: SparkSession, model_path: str = DEFAULT_RF_MODEL_PATH):
        self.spark = spark
        self.model_path = model_path
        logger.info(f"Loading PipelineModel from: {model_path}")
        self.model = PipelineModel.load(model_path)
        logger.info("PipelineModel loaded successfully.")

    def prepare_input_dataframe(self, car_features: Dict[str, Any]):
        """
        Transforms raw user input dict into the structured schema required by the ML pipeline.
        Derives vehicle_age, km_per_year, power_to_engine_ratio, and is_luxury_brand automatically.
        """
        year = int(car_features.get("year", 2017))
        vehicle_age = max(1, 2025 - year)
        km_driven = float(car_features.get("km_driven", 50000.0))
        km_per_year = round(km_driven / vehicle_age, 2)
        
        brand = str(car_features.get("brand", "Maruti")).strip()
        fuel = str(car_features.get("fuel", "Petrol")).strip()
        seller_type = str(car_features.get("seller_type", "Individual")).strip()
        transmission = str(car_features.get("transmission", "Manual")).strip()
        owner = str(car_features.get("owner", "First Owner")).strip()
        
        mileage_kmpl = float(car_features.get("mileage_kmpl", 19.0))
        engine_cc = float(car_features.get("engine_cc", 1197.0))
        max_power_bhp = float(car_features.get("max_power_bhp", 82.0))
        seats = int(car_features.get("seats", 5))

        power_to_engine_ratio = round(max_power_bhp / engine_cc, 4) if engine_cc > 0 else 0.06
        is_luxury_brand = 1 if brand.lower() in LUXURY_BRANDS else 0

        # Construct single-row DataFrame
        row_dict = {
            "brand": brand,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner,
            "vehicle_age": float(vehicle_age),
            "km_driven": float(km_driven),
            "mileage_kmpl_std": float(mileage_kmpl),
            "engine_cc": float(engine_cc),
            "max_power_bhp": float(max_power_bhp),
            "seats": int(seats),
            "km_per_year": float(km_per_year),
            "power_to_engine_ratio": float(power_to_engine_ratio),
            "is_luxury_brand": int(is_luxury_brand),
            "selling_price": 0.0  # Dummy label for pipeline schema compatibility
        }

        return self.spark.createDataFrame([row_dict])

    def predict(self, car_features: Dict[str, Any]) -> float:
        """
        Returns predicted vehicle selling price in INR.
        """
        df_input = self.prepare_input_dataframe(car_features)
        transformed = self.model.transform(df_input)
        prediction = transformed.select("prediction").collect()[0]["prediction"]
        return max(15000.0, float(prediction))
