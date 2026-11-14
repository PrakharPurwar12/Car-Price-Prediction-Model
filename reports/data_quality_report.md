# CarIntel — Data Quality & Ingestion Audit Report

**Generated Automatically by CarIntel PySpark ETL Pipeline**  
**Pipeline Status:** SUCCESS  

---

## 1. Executive Summary

The CarIntel Distributed ETL Pipeline ingested the original CarDekho dataset, executed strict schema validation, eliminated duplicate records to fix data leakage, parsed vehicle brand and model hierarchies, standardized mileage units, imputed impossible zero specifications, and engineered domain features.

### Record Accounting

| Stage | Metric Description | Value |
|---|---|---|
| **Raw Ingestion** | Initial Record Count | **8,148** |
| **Raw Ingestion** | Total Input Columns | **13** |
| **Deduplication** | Exact Duplicate Rows Identified | **1,202** |
| **Deduplication** | Unique Observations Retained | **6,946** |
| **Cleaning** | Corrupt / Non-positive Records Removed | **0** |
| **Output** | Final Valid Clean Records | **6,946** |
| **Output** | Final Feature Count | **28** |
| **Storage** | Target Format | **Apache Parquet (Snappy, Partitioned by Fuel)** |

---

## 2. Missing Value Analysis (Raw Dataset)

| Attribute Name | Null Count in Raw CSV |
|---|---|
| `Name` | 0 |
| `year` | 0 |
| `selling_price` | 0 |
| `km_driven` | 0 |
| `fuel` | 0 |
| `seller_type` | 0 |
| `transmission` | 0 |
| `owner` | 0 |
| `mileage` | 221 |
| `engine` | 221 |
| `max_power` | 215 |
| `torque` | 222 |
| `seats` | 221 |

---

## 3. Data Cleaning & Engineering Decisions

1. **Exact Deduplication:** Eliminated `1,202` duplicate rows. In Phase 1 audit, we proved that failure to remove duplicates caused severe train/test data leakage, as duplicate rows leaked into both sets and artificially elevated model performance.
2. **Multi-word Brand Disambiguation:** Sliced multi-word automotive brands (`Land Rover`, `Mercedes-Benz`, `Ashok Leyland`) before extracting model hierarchy, resolving previous `Land` and `Ashok` truncation bugs.
3. **Mileage Unit Normalization:** Extracted numerical value and preserved fuel unit (`kmpl` vs `km/kg`). Converted CNG/LPG `km/kg` to gasoline-equivalent energy density `km/l` ($1\text{ km/kg} \approx 1.39\text{ km/l}$).
4. **Outlier & Impossible Value Imputation:**
   - Replaced 17 physically impossible `mileage = 0.0 kmpl` records with median (19.30 km/l).
   - Replaced 6 `max_power = 0.0 bhp` records with median (82.00 bhp).
   - Imputed missing engine displacement with median (1248.0 CC).
5. **Target Transformation:** Computed $\ln(1 + \text{selling\_price})$ (`log_selling_price`) to handle right-skewness spanning ₹29,999 to ₹10,000,000.
6. **Feature Additions:** Engineered `vehicle_age`, `km_per_year`, `power_to_engine_ratio`, and `is_luxury_brand`.

---

## 4. Output Storage

The processed dataset is serialized to partitioned Apache Parquet at:
`F:\Prakhar\Prakhar\Projects\Car-Price-Prediction-Model\data\processed\vehicles.parquet`
