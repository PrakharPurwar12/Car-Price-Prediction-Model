# CarIntel — Distributed Machine Learning Evaluation Report

## 1. Executive Summary

This report documents the performance of distributed machine learning models developed using **Apache Spark MLlib** for the CarIntel platform. Models were trained on clean, deduplicated, standardized vehicle records (6946 verified vehicles) using an 80/20 deterministic train/test split.

---

## 2. Model Performance Matrix

| Metric | Legacy Scikit-Learn (Flawed Leakage) | Spark MLlib Random Forest | Spark MLlib Gradient-Boosted Trees |
|---|---|---|---|
| **Architecture** | Single-node (Scikit-Learn) | Distributed (Spark MLlib) | Distributed (Spark MLlib) |
| **Data Leakage Controlled?** | ❌ No (Duplicates Present) | ✅ Yes (1,202 Duplicates Removed) | ✅ Yes (1,202 Duplicates Removed) |
| **Train $R^2$** | 0.9740 | 0.962 | 0.9805 |
| **Test $R^2$** | 0.9128 | **0.9** | **0.7915** |
| **Test RMSE (₹)** | ~185,000 | ₹147,816.84 | ₹213,456.74 |
| **Test MAE (₹)** | ~98,000 | ₹81,699.35 | ₹88,211.31 |
| **Resilience to Unseen Brands (e.g. Peugeot)** | ❌ Crashes (`KeyError`) | ✅ Handled (`handleInvalid="keep"`) | ✅ Handled (`handleInvalid="keep"`) |

---

## 3. Analysis & Key Takeaways

1. **True Generalization vs. Artificial Inflation:**
   - The legacy baseline reported $R^2 = 0.9128$, but our Phase 1 audit revealed **1,202 duplicate records** leaked across train and test partitions.
   - The Spark MLlib Random Forest achieved a genuine **Test $R^2 = 0.9** with strictly zero data contamination.

2. **Feature Representation & Resilience:**
   - Used `StringIndexer` with `handleInvalid="keep"` and `OneHotEncoder` to encode categorical dimensions (`brand`, `fuel`, `seller_type`, `transmission`, `owner`).
   - Standardized mileage units across `kmpl` and `km/kg` into an energy-equivalent metric (`mileage_kmpl_std`).
   - Combined domain features (`power_to_engine_ratio`, `km_per_year`, `vehicle_age`, `is_luxury_brand`).

3. **Inference Readiness:**
   - Both models are persisted as full `PipelineModel` artifacts (`ml/models/rf_model`, `ml/models/gbt_model`), eliminating preprocessing-code duplication in production.

*Report generated automatically by CarIntel Model Evaluation Pipeline.*
