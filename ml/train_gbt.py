"""
ml/train_gbt.py
Distributed Gradient-Boosted Trees (GBTRegressor) training for CarIntel:
Trains a Spark MLlib GBTRegressor pipeline, records RMSE, MAE, R²,
and serializes the PipelineModel to disk.
"""

import json
import logging
from pathlib import Path

import config.spark_config
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from config.spark_config import get_spark_session
from ml.feature_pipeline import build_feature_pipeline, TARGET_COL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainGBT")

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = str(ROOT_DIR / "data" / "processed" / "vehicles.parquet")
MODEL_OUTPUT_DIR = str(ROOT_DIR / "ml" / "models" / "gbt_model")
METRICS_OUTPUT_PATH = ROOT_DIR / "ml" / "models" / "gbt_metrics.json"


def train_gradient_boosted_trees(spark):
    logger.info(f"Loading processed vehicle data from: {PROCESSED_DATA_PATH}")
    df = spark.read.parquet(PROCESSED_DATA_PATH)

    # Train / Test split (80/20) matching RF split
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    logger.info(f"Training set: {train_df.count()} rows | Test set: {test_df.count()} rows")

    feature_pipe = build_feature_pipeline(include_scaler=False)

    gbt = GBTRegressor(
        featuresCol="features",
        labelCol=TARGET_COL,
        maxIter=60,
        maxDepth=6,
        seed=42,
        stepSize=0.1
    )

    full_stages = feature_pipe.getStages() + [gbt]
    pipeline = Pipeline(stages=full_stages)

    logger.info("Fitting Spark MLlib GBTRegressor pipeline on distributed cluster...")
    model = pipeline.fit(train_df)

    evaluator_rmse = RegressionEvaluator(labelCol=TARGET_COL, predictionCol="prediction", metricName="rmse")
    evaluator_mae = RegressionEvaluator(labelCol=TARGET_COL, predictionCol="prediction", metricName="mae")
    evaluator_r2 = RegressionEvaluator(labelCol=TARGET_COL, predictionCol="prediction", metricName="r2")

    logger.info("Evaluating GBT model...")
    test_preds = model.transform(test_df)
    train_preds = model.transform(train_df)

    test_rmse = evaluator_rmse.evaluate(test_preds)
    test_mae = evaluator_mae.evaluate(test_preds)
    test_r2 = evaluator_r2.evaluate(test_preds)

    train_rmse = evaluator_rmse.evaluate(train_preds)
    train_mae = evaluator_mae.evaluate(train_preds)
    train_r2 = evaluator_r2.evaluate(train_preds)

    logger.info("=" * 60)
    logger.info(f"GBT Train R²: {train_r2:.4f} | Train RMSE: {train_rmse:.2f} | Train MAE: {train_mae:.2f}")
    logger.info(f"GBT Test  R²: {test_r2:.4f} | Test  RMSE: {test_rmse:.2f} | Test  MAE: {test_mae:.2f}")
    logger.info("=" * 60)

    gbt_model = model.stages[-1]
    importances = gbt_model.featureImportances.toArray().tolist()

    metrics = {
        "model_type": "Spark MLlib GBTRegressor",
        "parameters": {
            "maxIter": gbt_model.getMaxIter(),
            "maxDepth": gbt_model.getMaxDepth(),
            "stepSize": gbt_model.getStepSize(),
            "seed": 42
        },
        "train_metrics": {
            "r2": round(train_r2, 4),
            "rmse": round(train_rmse, 2),
            "mae": round(train_mae, 2)
        },
        "test_metrics": {
            "r2": round(test_r2, 4),
            "rmse": round(test_rmse, 2),
            "mae": round(test_mae, 2)
        },
        "feature_importances_count": len(importances)
    }

    METRICS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"GBT Metrics saved to: {METRICS_OUTPUT_PATH}")

    logger.info(f"Saving Spark MLlib GBT PipelineModel to: {MODEL_OUTPUT_DIR}")
    model.write().overwrite().save(MODEL_OUTPUT_DIR)
    logger.info("GBT PipelineModel saved successfully.")

    return model, metrics


if __name__ == "__main__":
    spark = get_spark_session("CarIntel-TrainGBT")
    try:
        train_gradient_boosted_trees(spark)
    finally:
        spark.stop()
