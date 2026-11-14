"""
ml/feature_pipeline.py
Constructs a reusable Spark MLlib feature transformation pipeline:
- StringIndexer with handleInvalid="keep" (resilient to unseen brands like Peugeot)
- OneHotEncoder for indexed categorical features
- VectorAssembler combining encoded categorical features + normalized numeric features
"""

import config.spark_config
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)

CATEGORICAL_COLS = ["brand", "fuel", "seller_type", "transmission", "owner"]
NUMERICAL_COLS = [
    "vehicle_age",
    "km_driven",
    "mileage_kmpl_std",
    "engine_cc",
    "max_power_bhp",
    "seats",
    "km_per_year",
    "power_to_engine_ratio",
    "is_luxury_brand",
]
TARGET_COL = "selling_price"
LOG_TARGET_COL = "log_selling_price"


def build_feature_pipeline(include_scaler: bool = False) -> Pipeline:
    """
    Builds a PySpark ML Pipeline for feature preprocessing.
    Uses handleInvalid='keep' to gracefully handle previously unseen categories.
    """
    stages = []

    # 1. StringIndexer for each categorical feature
    indexed_cols = [f"{col}_idx" for col in CATEGORICAL_COLS]
    encoded_cols = [f"{col}_vec" for col in CATEGORICAL_COLS]

    for cat_col, idx_col in zip(CATEGORICAL_COLS, indexed_cols):
        indexer = StringIndexer(
            inputCol=cat_col,
            outputCol=idx_col,
            handleInvalid="keep"
        )
        stages.append(indexer)

    # 2. OneHotEncoder
    encoder = OneHotEncoder(
        inputCols=indexed_cols,
        outputCols=encoded_cols,
        handleInvalid="keep"
    )
    stages.append(encoder)

    # 3. VectorAssembler
    assembler_inputs = encoded_cols + NUMERICAL_COLS
    assembler = VectorAssembler(
        inputCols=assembler_inputs,
        outputCol="raw_features",
        handleInvalid="keep"
    )
    stages.append(assembler)

    # 4. Optional StandardScaler
    if include_scaler:
        scaler = StandardScaler(
            inputCol="raw_features",
            outputCol="features",
            withStd=True,
            withMean=False
        )
        stages.append(scaler)
    else:
        # Pass-through assembler directly to 'features'
        stages[-1] = VectorAssembler(
            inputCols=assembler_inputs,
            outputCol="features",
            handleInvalid="keep"
        )

    pipeline = Pipeline(stages=stages)
    return pipeline
