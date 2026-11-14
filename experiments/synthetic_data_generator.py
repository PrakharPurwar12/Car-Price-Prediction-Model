"""
experiments/synthetic_data_generator.py
Generates deterministically scaled vehicle datasets (1x, 5x, 10x, 20x)
by perturbing genuine records while preserving domain statistics,
correlation structures, and realistic automotive constraints.
"""

import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SyntheticDataGenerator")

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_CSV_PATH = ROOT_DIR / "data" / "raw" / "Car dekho - Car dekho.csv"
SYNTHETIC_DIR = ROOT_DIR / "data" / "synthetic"


def generate_scaled_datasets(scales=(1, 5, 10, 20)):
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {RAW_CSV_PATH}")

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading base dataset from: {RAW_CSV_PATH}")
    base_df = pd.read_csv(RAW_CSV_PATH)
    base_count = len(base_df)
    logger.info(f"Base dataset record count: {base_count}")

    np.random.seed(42)

    generated_files = {}

    for scale in scales:
        target_rows = base_count * scale
        output_file = SYNTHETIC_DIR / f"vehicles_{scale}x.csv"

        if scale == 1:
            base_df.to_csv(output_file, index=False)
            generated_files[f"{scale}x"] = {
                "path": str(output_file),
                "rows": base_count,
                "size_mb": round(output_file.stat().st_size / (1024 * 1024), 2)
            }
            logger.info(f"Generated 1x baseline: {base_count} rows ({generated_files['1x']['size_mb']} MB)")
            continue

        logger.info(f"Generating {scale}x dataset ({target_rows} rows)...")

        # Sample with replacement
        indices = np.random.choice(base_count, size=target_rows, replace=True)
        synthetic_df = base_df.iloc[indices].copy().reset_index(drop=True)

        # Introduce realistic perturbation to numerical columns
        # Year perturbation: +/- 1 year bounded
        year_jitter = np.random.choice([-1, 0, 1], size=target_rows, p=[0.25, 0.50, 0.25])
        synthetic_df["year"] = np.clip(synthetic_df["year"] + year_jitter, 1995, 2024)

        # km_driven perturbation: +/- 5%
        km_factor = 1.0 + np.random.normal(0, 0.05, size=target_rows)
        synthetic_df["km_driven"] = np.clip((synthetic_df["km_driven"] * km_factor).astype(int), 500, 2500000)

        # selling_price perturbation: +/- 8%
        price_factor = 1.0 + np.random.normal(0, 0.08, size=target_rows)
        synthetic_df["selling_price"] = np.clip((synthetic_df["selling_price"] * price_factor).astype(int), 20000, 15000000)

        synthetic_df.to_csv(output_file, index=False)
        size_mb = round(output_file.stat().st_size / (1024 * 1024), 2)
        generated_files[f"{scale}x"] = {
            "path": str(output_file),
            "rows": len(synthetic_df),
            "size_mb": size_mb
        }
        logger.info(f"Generated {scale}x dataset: {len(synthetic_df)} rows ({size_mb} MB) -> {output_file.name}")

    return generated_files


if __name__ == "__main__":
    generate_scaled_datasets(scales=(1, 5, 10, 20))

