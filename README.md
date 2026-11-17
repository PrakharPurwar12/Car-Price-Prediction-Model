# CarIntel — Distributed Vehicle Market Intelligence & Resale Valuation Platform

[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-4.2.0-E25A1C.svg?style=flat&logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![Spark MLlib](https://img.shields.io/badge/Spark-MLlib-E25A1C.svg?style=flat)](https://spark.apache.org/mllib/)
[![Spark GraphX](https://img.shields.io/badge/Spark-GraphX-E25A1C.svg?style=flat)](https://spark.apache.org/graphx/)
[![Scala](https://img.shields.io/badge/Scala-2.13.18-DC322F.svg?style=flat&logo=scala&logoColor=white)](https://www.scala-lang.org/)
[![Java](https://img.shields.io/badge/Java-17%20LTS-007396.svg?style=flat&logo=openjdk&logoColor=white)](https://openjdk.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Apache Parquet](https://img.shields.io/badge/Storage-Parquet%20Snappy-5B92E5.svg?style=flat)](https://parquet.apache.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Tests-12%20Passed%20(100%25)-brightgreen.svg?style=flat&logo=pytest&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat)](LICENSE)
[![Academic Alignment](https://img.shields.io/badge/Course-INT315%20Cluster%20Computing-1B365D.svg?style=flat)]()

> **CarIntel** is an enterprise-grade distributed computing platform for automotive market intelligence, topological graph analytics, and real-time resale price prediction. Built on **Apache Spark 4.2.0**, **Spark SQL**, **Scala GraphX**, and **Spark MLlib**, CarIntel replaces heuristic single-node scripts with partitioned out-of-core pipelines, decoupled columnar Parquet persistence, and zero-crash production inference.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Ingestion ["1. Distributed Ingestion & Hygiene"]
        RawCSV[("Raw Dataset<br/>Car dekho.csv<br/>(8,148 rows)")] --> SparkIngest["pyspark.sql.DataFrameReader<br/>Mode: DROPMALFORMED<br/>Schema: RAW_CAR_SCHEMA"]
        SparkIngest --> DedupCheck["Exact Deduplication<br/>Removed 1,202 Duplicates (14.75%)"]
    end

    subgraph Preprocessing ["2. Distributed Preprocessing & Feature Engineering"]
        DedupCheck --> BrandExtractor["Multi-Word Brand & Model Parser<br/>(Land Rover, Mercedes-Benz, Ashok Leyland)"]
        BrandExtractor --> UnitParser["Regex Normalizer & Standardizer<br/>mileage (kmpl & km/kg) -> mileage_kmpl_std<br/>engine (CC) -> engine_cc<br/>max_power (bhp) -> max_power_bhp"]
        UnitParser --> Imputer["Deterministic Median Imputation<br/>mileage: 19.3 kmpl | engine: 1248 CC | power: 82 bhp"]
        Imputer --> FeatEng["Automotive Feature Engineering<br/>vehicle_age, km_per_year, power_to_engine_ratio, is_luxury"]
        FeatEng --> ParquetOut[("Optimized Columnar Parquet<br/>data/processed/vehicles.parquet<br/>(6,946 clean records)")]
    end

    subgraph AnalyticsEngines ["3. Analytical & Machine Learning Engines"]
        ParquetOut --> SparkSQL["Apache Spark SQL<br/>Whole-Stage Code Generation<br/>6 Aggregated Market Queries"]
        ParquetOut --> GraphX["Scala Spark GraphX<br/>Heterogeneous Bipartite Graph<br/>PageRank & Connected Components"]
        ParquetOut --> MLlib["Spark MLlib Pipeline<br/>StringIndexer(keep) + OneHotEncoder + VectorAssembler<br/>RandomForestRegressor (R² = 0.9000)"]
    end

    subgraph Presentation ["4. Delivery & Interactive Interface"]
        SparkSQL --> Page1["Page 1: Market Overview & SQL Analytics"]
        GraphX --> Page2["Page 2: Graph Intelligence & Market Hubs"]
        MLlib --> Page3["Page 3: Real-Time Valuation & Inference"]
        ParquetOut --> Page4["Page 4: Empirical Scalability Benchmarks"]
        Presentation --> DocGen["Academic Report Generator<br/>generate_report.py -> DOCX (A4, IEEE)"]
    end

    style Ingestion fill:#f0f7ff,stroke:#0366d6,stroke-width:2px
    style Preprocessing fill:#fef9e7,stroke:#f5b041,stroke-width:2px
    style AnalyticsEngines fill:#fdf2e9,stroke:#e67e22,stroke-width:2px
    style Presentation fill:#eafaf1,stroke:#2ecc71,stroke-width:2px
```

---

## 🌟 Key Engineering Innovations

### 1. Windows Native Hadoop Storage Engine (`WinLocalFileSystem`)
Standard Apache Spark installations on Windows rely on `winutils.exe` to execute POSIX `chmod` operations during Parquet partition writes. In modern Windows environments with Application Control (AppLocker), external binaries trigger blocking errors (`exit code 4551`).
- **Solution:** Authored `com.carintel.hadoop.WinLocalFileSystem` in pure Java 17, extending Hadoop's `LocalFileSystem` and `RawLocalFileSystem`.
- **Impact:** Uses native `java.io.File` directory operations and URI status resolution, enabling native Parquet I/O on Windows with **strictly zero dependencies on `winutils.exe`**.

### 2. Elimination of Train/Test Data Leakage
The raw CarDekho dataset contains **1,202 exact duplicate records (14.75% of raw data)**. In naive random-split workflows, identical records land in both training and test partitions, artificially inflating model accuracy ($R^2 = 0.9128$).
- **Solution:** Distributed exact deduplication via `dropDuplicates()` inside the PySpark ETL stage before partitioning.
- **Impact:** Yields 6,946 verified clean listings and a legitimate test $R^2 = 0.9000$.

### 3. Crash-Proof Inference on Unseen Categories
Legacy single-node implementations failed in production when encountering previously unobserved categorical values (e.g., brand `'Peugeot'` raising unhandled `KeyError: 19`).
- **Solution:** Spark MLlib `StringIndexer` configured with `handleInvalid="keep"`.
- **Impact:** Dynamically assigns unseen categorical values to a dedicated reserve index bucket, ensuring zero-crash runtime resilience in production inference.

### 4. Bipartite Multiplex Graph Intelligence
Tabular SQL cannot easily discover recursive structural prestige or brand community hubs across multi-hop relationships.
- **Solution:** Scala Spark GraphX builds a heterogeneous property graph with 6,985 vertices (`VEHICLE`, `BRAND`, `FUEL`, `TRANSMISSION`) and 20,838 directed edges (`MANUFACTURED_BY`, `USES_FUEL`, `HAS_TRANSMISSION`).
- **Impact:** Pregel-based distributed PageRank identifies market centrality anchors (e.g., Manual Transmission PageRank `976.33`, Diesel `577.88`, Maruti `333.59`).

---

## 📊 Requirement-to-Implementation Matrix

| Requirement | Legacy Baseline Implementation | CarIntel Distributed Platform | Verification |
|---|---|---|---|
| **System Architecture** | Single-script `app.py` & Jupyter notebook | Decoupled modular architecture: compute, storage, graph & ML layers | ✅ **Verified** |
| **Distributed Ingestion** | In-memory `pandas.read_csv` | `pyspark.sql.DataFrameReader` with explicit `StructType` schema & `DROPMALFORMED` | ✅ **Verified** |
| **Data Hygiene & ETL** | Ignored 1,202 duplicate records (data leakage) | Exact deduplication, multi-word brand extraction, energy-equivalent mileage units | ✅ **Verified** |
| **Optimized Storage** | Raw CSV files | Snappy-compressed Columnar **Apache Parquet** (`data/processed/vehicles.parquet`) | ✅ **Verified** |
| **Analytical Queries** | Basic in-memory groupbys | **Apache Spark SQL** Whole-Stage Code Generation executing 6 aggregated analytical views | ✅ **Verified** |
| **Graph Analytics** | None | **Scala Spark GraphX** (6,985 vertices, 20,838 edges, PageRank, Connected Components) | ✅ **Verified** |
| **Distributed ML** | Scikit-learn on single node (leakage-inflated $R^2 = 0.9128$) | **Spark MLlib** `RandomForestRegressor` ($R^2 = 0.9000$) & `GBTRegressor` ($R^2 = 0.7915$) | ✅ **Verified** |
| **Edge-Case Resilience** | Crashes on unseen brands (`KeyError: 19`) | `StringIndexer(handleInvalid="keep")` absorbs unseen categories gracefully | ✅ **Verified** |
| **Scalability Benchmarks** | None | Empirical scalability benchmarks across **1×, 5×, 10×, 20× scales** (up to 162,960 records) | ✅ **Verified** |
| **Interactive Dashboard** | Single-page script with hardcoded charts | 4-page responsive Streamlit dashboard with cached data loaders & live inference | ✅ **Verified** |
| **Academic Report** | Static unstructured documentation | `generate_report.py` producing publication-grade IEEE-formatted DOCX report | ✅ **Verified** |

---

## 🔬 Machine Learning Evaluation

Evaluated on clean, deduplicated data using a deterministic 80/20 train/test split (seed: `42`):

| Model Architecture | Train $R^2$ | Test $R^2$ | Test RMSE | Test MAE | Data Leakage Controlled? | Unseen Category Resilience |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Legacy Scikit-Learn RF** | 0.9740 | 0.9128* | ~₹185,000 | ~₹98,000 | ❌ No (1,202 duplicates leaked) | ❌ Crashes (`KeyError`) |
| **Spark MLlib Random Forest** | **0.9620** | **0.9000** | **₹147,816** | **₹81,699** | ✅ **Yes (Zero Duplicates)** | ✅ **Resilient (`keep`)** |
| **Spark MLlib GBT** | 0.9805 | 0.7915 | ₹213,456 | ₹88,211 | ✅ **Yes (Zero Duplicates)** | ✅ **Resilient (`keep`)** |

*\*Note: Legacy $R^2 = 0.9128$ was artificially inflated due to identical duplicate records present across training and test partitions.*

### Dominant Price Drivers (Spark MLlib Feature Importance)
1. **`max_power_bhp`**: Peak engine brake horsepower output (highest relative importance)
2. **`engine_cc`**: Total engine displacement in cubic centimeters
3. **`vehicle_age`**: Vehicle age computed dynamically from `REFERENCE_YEAR` (2025)

---

## ⚡ Empirical Scalability Benchmarks

Benchmarked across synthetically scaled datasets on a Windows 11 development cluster (`local[*]` master):

| Scale Factor | Raw Records | Pandas ETL (s) | Spark ETL (s) | Spark Throughput (rec/s) | Sklearn Train (s) | Spark MLlib Train (s) | Spark SQL Query (s) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1×** | 8,148 | 0.052s | 7.557s | 919.2 | 0.192s | 8.081s | 0.482s |
| **5×** | 40,740 | 0.269s | 3.416s | 11,924.5 | 0.143s | 7.132s | 0.431s |
| **10×** | 81,480 | 0.628s | 3.162s | 25,765.1 | 0.177s | 8.943s | 0.400s |
| **20×** | 162,960 | 1.029s | 4.616s | **35,300.2** | 0.183s | 13.334s | 0.788s |

> **Key Distributed Systems Insight:** At small data volumes (1×), single-node tools appear faster due to Spark's JVM startup, Catalyst compilation, and task serialization overhead. However, as dataset size expands to 20×, Spark's throughput surges from **919 to 35,300+ records/sec (38.4× increase)**, confirming horizontal scaling capability and immunity to single-node memory exhaustion.

---

## 🕸️ Scala Spark GraphX Topology Analysis

CarIntel constructs a bipartite multiplex graph connecting vehicle listings to attribute anchors:
- **Vertices (6,985 total):** 6,946 vehicle listings, 32 brand hubs, 5 fuel nodes, 2 transmission nodes.
- **Edges (20,838 total):** `MANUFACTURED_BY`, `USES_FUEL`, `HAS_TRANSMISSION`.
- **Connected Components:** Exactly 1 (fully connected graph).

### Top PageRank Centrality Anchors
1. **Manual Transmission:** PageRank `976.33` (Degree: 6,355)
2. **Diesel Fuel:** PageRank `577.88` (Degree: 3,760)
3. **Petrol Fuel:** PageRank `475.16` (Degree: 3,091)
4. **Maruti Brand Hub:** PageRank `333.59` (Degree: 2,169)
5. **Hyundai Brand Hub:** PageRank `195.70` (Degree: 1,271)
6. **Mahindra Brand Hub:** PageRank `111.86` (Degree: 725)
7. **Tata Brand Hub:** PageRank `100.04` (Degree: 648)
8. **Automatic Transmission:** PageRank `91.29` (Degree: 591)

---

## 📁 Repository Structure

```
CarIntel/
├── config/
│   ├── spark_config.py          # Unified SparkSession factory with Java 17 module flags
│   ├── src/com/carintel/hadoop/ # Custom Java WinLocalFileSystem source
│   └── win_local_fs.jar         # Compiled pure-Java Windows Hadoop bypass engine
├── dashboard/                   # Multi-Page Streamlit Dashboard
│   ├── streamlit_app.py         # Application entry point & architecture overview
│   ├── utils.py                 # Cached data loaders & formatters
│   └── pages/
│       ├── 1_Market_Overview.py         # Spark SQL brand & fuel analytics
│       ├── 2_Graph_Intelligence.py      # GraphX PageRank & network analysis
│       ├── 3_Price_Prediction.py        # Real-time Spark MLlib inference
│       └── 4_Scalability_Benchmarks.py  # Empirical throughput curves
├── data/
│   ├── raw/                     # Original preserved raw CSV dataset
│   ├── processed/               # Cleaned columnar Parquet (vehicles.parquet)
│   ├── graph_metrics/           # GraphX PageRank & Connected Components metrics
│   └── synthetic/               # Scaled benchmark datasets (1x, 5x, 10x, 20x)
├── experiments/                 # Empirical Scalability Suite
│   ├── synthetic_data_generator.py # Deterministic multi-scale data generator
│   ├── benchmark_runner.py         # Timed comparative benchmark orchestrator
│   └── results/                    # Benchmark CSV & JSON metrics
├── graphx/                      # Apache Spark GraphX (Scala)
│   ├── build.sbt                # SBT build definition (Scala 2.13.18, Spark 4.2.0)
│   ├── src/main/scala/com/carintel/graph/
│   │   ├── VehicleGraphBuilder.scala     # Multiplex graph construction
│   │   ├── GraphAnalyticsRunner.scala    # PageRank & Connected Components
│   │   └── GraphMetricsExporter.scala    # Parquet/JSON metric exporter
│   └── target/carintel-graphx.jar        # Pre-compiled GraphX JAR
├── legacy/                      # Preserved original baseline implementation
│   ├── app.py                   # Original single-node Streamlit script
│   └── main.ipynb               # Original exploratory notebook
├── ml/                          # Distributed Machine Learning (Spark MLlib)
│   ├── feature_pipeline.py      # StringIndexer(keep) + OneHotEncoder + VectorAssembler
│   ├── train_rf.py              # RandomForestRegressor distributed training
│   ├── train_gbt.py             # GBTRegressor distributed training
│   ├── evaluate.py              # Comparative evaluation & metrics export
│   ├── predict.py               # Production inference engine with auto feature derivation
│   └── models/                  # Serialized PipelineModel artifacts
├── reports/                     # Academic figures & report artifacts
│   └── figures/                 # 11 execution & dashboard screenshots (PNG)
├── scripts/                     # PowerShell & automation scripts
│   ├── compile_graphx.py        # Embedded Scala GraphX compiler
│   ├── run_graphx.py            # Spark GraphX execution runner
│   ├── generate_graphx_visualizations.py # Publication network visualizer
│   ├── run_etl.ps1              # Distributed ETL execution script
│   ├── run_training.ps1         # MLlib training pipeline script
│   └── run_dashboard.ps1        # Interactive Streamlit launch script
├── spark_processing/            # Distributed PySpark ETL & Analytics Pipeline
│   ├── schemas.py               # Explicit StructType schema definitions
│   ├── ingestion.py             # Schema-enforced ingestion & duplicate audit
│   ├── preprocessing.py         # Deduplication, regex parsing & median imputation
│   ├── feature_engineering.py   # Domain automotive feature derivation
│   ├── etl_pipeline.py          # End-to-end distributed ETL runner
│   └── analytical_queries.py    # Spark SQL analytical views & KPI computations
├── tests/                       # Automated Pytest Suite (12 tests)
│   ├── test_spark_session.py    # Session factory & configuration tests
│   ├── test_data_validation.py  # Schema contracts & invariant checks
│   ├── test_etl.py              # Deduplication & imputation tests
│   └── test_ml_pipeline.py      # Pipeline smoke & unseen category resilience tests
├── visualizations/              # Network graph figures (PNG)
│   └── graphx/                  # PageRank, Connected Components & topology plots
├── CarIntel_INT315_Project_Report.docx # Publication-grade academic project report
├── generate_report.py           # Report generator using python-docx
├── pytest.ini                   # Pytest discovery configuration
├── requirements.txt             # Unified Python dependencies
└── LICENSE                      # MIT Open Source License
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Prerequisites
- **Java 17 LTS** (configured in `JAVA_HOME` or installed at `C:\Program Files\Java\jdk-17`)
- **Apache Spark 4.2.0** (configured in `SPARK_HOME` or installed at `C:\spark\spark-4.2.0-bin-hadoop3`)
- **Python 3.10+** (tested on Python 3.10 – 3.14)

### 2. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/PrakharPurwar12/Car-Price-Prediction-Model.git
cd Car-Price-Prediction-Model

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Distributed ETL & Spark SQL Analytics
```powershell
python -m spark_processing.etl_pipeline
python -m spark_processing.analytical_queries
```

### 4. Compile & Execute Scala GraphX Pipeline
```powershell
python scripts/compile_graphx.py
python scripts/run_graphx.py
```

### 5. Train Spark MLlib Machine Learning Models
```powershell
python -m ml.train_rf
python -m ml.train_gbt
python -m ml.evaluate
```

### 6. Run Empirical Scalability Benchmarks
```powershell
python -m experiments.synthetic_data_generator
python -m experiments.benchmark_runner
```

### 7. Run Automated Pytest Suite
```powershell
pytest tests/ -v
```

### 8. Launch Interactive Streamlit Dashboard
```powershell
streamlit run dashboard/streamlit_app.py
```

### 9. Generate Academic Project Report (DOCX)
```powershell
python generate_report.py
```
Outputs `CarIntel_INT315_Project_Report.docx` formatted strictly to university standards:
- All body text in **Times New Roman, 13pt, Black, Justified**
- Headings in **16pt** (H1) and **14.5pt** (H2)
- Accurately synchronized **Table of Contents** with dot leaders
- All 11 colour execution and output screenshots embedded cleanly
- Hard page budget strictly controlled (well under the 50-page maximum limit)

---

## 🎓 Viva & Presentation Talking Points

1. **Why Apache Spark over Pandas for this platform?**
   - Pandas is bound to a single thread and constrained by the Python Global Interpreter Lock (GIL); large datasets trigger unrecoverable `OutOfMemory` exceptions.
   - Spark executes partitioned out-of-core operations with Catalyst query optimization, Whole-Stage Code Generation, and Tungsten off-heap memory management.

2. **What was the data leakage issue and how was it solved?**
   - 1,202 duplicate listings (14.75% of raw data) existed in the original dataset. Naive splitting placed identical records into both training and evaluation subsets, artificially inflating test metrics.
   - CarIntel applies distributed exact deduplication before partitioning, yielding an authentic test $R^2 = 0.9000$.

3. **What is the significance of the `WinLocalFileSystem` Java innovation?**
   - On Windows, Spark defaults to Hadoop committers that call `winutils.exe chmod`, which fails in enterprise environments governed by AppLocker.
   - Authoring `com.carintel.hadoop.WinLocalFileSystem` overrides Hadoop's filesystem layer with native Java 17 I/O, completely eliminating the need for `winutils.exe`.

4. **How does GraphX add value beyond standard SQL queries?**
   - Standard SQL aggregations cannot discover multi-hop recursive structural prestige or graph centrality.
   - GraphX uses Pregel message-passing to compute PageRank across bipartite vehicle-brand-fuel-transmission graphs, uncovering market anchors invisible to tabular queries.

5. **How was the Peugeot production crash resolved?**
   - In legacy code, unseen brands during inference raised `KeyError: 19`.
   - CarIntel uses Spark MLlib's `StringIndexer(handleInvalid="keep")`, allocating previously unseen categories to a dedicated reserve bucket so the inference engine never crashes in production.

---

## 📚 References

1. M. Zaharia, R. S. Xin, P. Wendell, T. Das, M. Armbrust, A. Dave, X. Meng, J. Rosen, S. Venkataraman, M. J. Franklin, A. Ghodsi, J. Gonzalez, S. Shenker, and I. Stoica, "Apache Spark: A Unified Engine for Big Data Processing," *Communications of the ACM*, vol. 59, no. 11, pp. 56–65, 2016.
2. M. Armbrust, R. S. Xin, C. Lian, Y. Huai, D. Liu, J. K. Bradley, X. Meng, T. Kaftan, M. J. Franklin, A. Ghodsi, and M. Zaharia, "Spark SQL: Relational Data Processing in Spark," in *Proc. 2015 ACM SIGMOD Int. Conf. Management of Data (SIGMOD '15)*, pp. 1383–1394, 2015.
3. R. S. Xin, J. E. Gonzalez, M. J. Franklin, and I. Stoica, "GraphX: A Resilient Distributed Graph System on Spark," in *Proc. First Int. Workshop on Graph Data Management Experiences and Systems (GRADES '13)*, Article 2, pp. 1–6, 2013.
4. X. Meng, J. Bradley, B. Yavuz, E. Sparks, S. Venkataraman, D. Liu, J. Freeman, D. Tsai, M. Amde, S. Owen, D. Xin, R. Xin, M. J. Franklin, R. Zadeh, M. Zaharia, and A. Talwalkar, "MLlib: Machine Learning in Apache Spark," *Journal of Machine Learning Research*, vol. 17, no. 34, pp. 1–7, 2016.
5. Kaggle, "Vehicle Dataset from CarDekho," Available: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho

---

## 📜 License & Acknowledgments

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.  
Developed for **INT315 — Cluster Computing**, Lovely Professional University.
