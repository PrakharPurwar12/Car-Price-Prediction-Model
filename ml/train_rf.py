"""
ml/train_rf.py
Distributed Machine Learning training for CarIntel:
Trains a Spark MLlib RandomForestRegressor pipeline with evaluation metrics (RMSE, MAE, R²),
extracts and saves feature importances, and serializes the complete PipelineModel.
"""

import json
import logging
import os
from pathlib import Path

import config.spark_config
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

from config.spark_config import get_spark_session
from ml.feature_pipeline import build_feature_pipeline, TARGET_COL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainRF")

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = str(ROOT_DIR / "data" / "processed" / "vehicles.parquet")
MODEL_OUTPUT_DIR = str(ROOT_DIR / "ml" / "models" / "rf_model")
METRICS_OUTPUT_PATH = ROOT_DIR / "ml" / "models" / "rf_metrics.json"


def train_random_forest(spark, run_cv: bool = False):
    logger.info(f"Loading processed vehicle data from: {PROCESSED_DATA_PATH}")
    df = spark.read.parquet(PROCESSED_DATA_PATH)
    logger.info(f"Total dataset records: {df.count()}")

    # Deterministic train/test split (80/20) on deduplicated data
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    logger.info(f"Training set: {train_df.count()} rows | Test set: {test_df.count()} rows")

    # Feature transformation pipeline
    feature_pipe = build_feature_pipeline(include_scaler=False)

    # Random Forest Regressor
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol=TARGET_COL,
        numTrees=100,
        maxDepth=12,
        seed=42,
        subsamplingRate=0.8
    )

    full_stages = feature_pipe.getStages() + [rf]
    pipeline = Pipeline(stages=full_stages)

    evaluator_rmse = RegressionEvaluator(labelCol=TARGET_COL, predictionCol="prediction", metricName="rmse")
    evaluator_mae = RegressionEvaluator(labelCol=TARGET_COL, predictionCol="prediction", metricName="mae")
    evaluator_r2 = RegressionEvaluator(labelCol=TARGET_COL, predictionCol="prediction", metricName="r2")

    if run_cv:
        logger.info("Setting up 3-fold Cross Validation grid...")
        param_grid = (
            ParamGridBuilder()
            .addGrid(rf.numTrees, [50, 100])
            .addGrid(rf.maxDepth, [8, 12])
            .build()
        )
        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator_r2,
            numFolds=3,
            seed=42
        )
        logger.info("Executing CrossValidator on distributed cluster...")
        cv_model = cv.fit(train_df)
        model = cv_model.bestModel
    else:
        logger.info("Fitting Spark MLlib RandomForestRegressor pipeline...")
        model = pipeline.fit(train_df)

    logger.info("Evaluating model on Test Set...")
    test_predictions = model.transform(test_df)
    train_predictions = model.transform(train_df)

    test_rmse = evaluator_rmse.evaluate(test_predictions)
    test_mae = evaluator_mae.evaluate(test_predictions)
    test_r2 = evaluator_r2.evaluate(test_predictions)

    train_rmse = evaluator_rmse.evaluate(train_predictions)
    train_mae = evaluator_mae.evaluate(train_predictions)
    train_r2 = evaluator_r2.evaluate(train_predictions)

    logger.info("=" * 60)
    logger.info(f"Train R²: {train_r2:.4f} | Train RMSE: {train_rmse:.2f} | Train MAE: {train_mae:.2f}")
    logger.info(f"Test  R²: {test_r2:.4f} | Test  RMSE: {test_rmse:.2f} | Test  MAE: {test_mae:.2f}")
    logger.info("=" * 60)

    # Extract feature importances
    rf_model = model.stages[-1]
    importances = rf_model.featureImportances.toArray().tolist()

    metrics = {
        "model_type": "Spark MLlib RandomForestRegressor",
        "parameters": {
            "numTrees": rf_model.getNumTrees,
            "maxDepth": rf_model.getMaxDepth(),
            "train_test_split": "80/20",
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

    # Save metrics JSON
    METRICS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {METRICS_OUTPUT_PATH}")

    # Save fitted PipelineModel
    logger.info(f"Saving Spark MLlib PipelineModel to: {MODEL_OUTPUT_DIR}")
    model.write().overwrite().save(MODEL_OUTPUT_DIR)
    logger.info("PipelineModel saved successfully.")

    return model, metrics


if __name__ == "__main__":
    spark = get_spark_session("CarIntel-TrainRF")
    try:
        train_random_forest(spark, run_cv=False)
    finally:
        spark.stop()
