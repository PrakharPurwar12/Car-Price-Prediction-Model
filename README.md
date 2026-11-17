# CarIntel — Distributed Vehicle Market Intelligence & Price Prediction Platform

[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-4.2.0-E25A1C.svg?style=flat&logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![Spark MLlib](https://img.shields.io/badge/Spark-MLlib-E25A1C.svg?style=flat)](https://spark.apache.org/mllib/)
[![Spark GraphX](https://img.shields.io/badge/Spark-GraphX-E25A1C.svg?style=flat)](https://spark.apache.org/graphx/)
[![Scala](https://img.shields.io/badge/Scala-2.13.18-DC322F.svg?style=flat&logo=scala&logoColor=white)](https://www.scala-lang.org/)
[![Java](https://img.shields.io/badge/Java-17%20LTS-007396.svg?style=flat&logo=openjdk&logoColor=white)](https://openjdk.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Tests-12%20Passed-brightgreen.svg?style=flat&logo=pytest&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat)](LICENSE)

---

## 📌 Project Description

**CarIntel** is a distributed big data analytics and machine learning platform engineered to solve vehicle valuation complexity and market information asymmetry in the secondary automotive sector. 

Operating on Apache Spark 4.2.0 with a decoupled storage architecture, CarIntel ingests raw vehicle listings into optimized columnar Snappy-compressed Apache Parquet datasets, runs multi-dimensional analytical queries via Spark SQL, constructs a heterogeneous knowledge graph using Scala GraphX to calculate market prestige metrics via PageRank, and trains resilient distributed regression models with Spark MLlib for real-time resale price prediction.

### Key Capabilities
- **Distributed ETL & Data Hygiene:** Enforces explicit StructType schemas, standardizes mixed-unit attributes (e.g. CNG fuel energy equivalent to kmpl), applies deterministic median imputation, and eliminates exact duplicates to prevent train/test data leakage.
- **Topological Graph Intelligence:** Models vehicle listings, brands, fuel types, and transmissions as a bipartite multiplex graph in Spark GraphX (6,985 vertices, 20,838 edges) to compute structural centrality and PageRank anchors.
- **Resilient Distributed ML:** Employs Spark MLlib `RandomForestRegressor` with `StringIndexer(handleInvalid="keep")` to guarantee zero-crash execution on previously unobserved vehicle categories.
- **Windows Hadoop Bypass Engine (`WinLocalFileSystem`):** Custom pure-Java filesystem extension that eliminates the requirement for `winutils.exe` on Windows environments.

---

## 🏛️ System Architecture Flowchart

```mermaid
flowchart TD
    subgraph S1 ["1. Distributed Ingestion & Hygiene"]
        RawCSV[("Raw Dataset<br/>Car dekho.csv<br/>(8,148 rows)")] --> SparkIngest["pyspark.sql.DataFrameReader<br/>Mode: DROPMALFORMED<br/>Schema: RAW_CAR_SCHEMA"]
        SparkIngest --> DedupCheck["Exact Deduplication<br/>Removed 1,202 Duplicates (14.75%)"]
    end

    subgraph S2 ["2. Preprocessing & Feature Engineering"]
        DedupCheck --> BrandExtractor["Multi-Word Brand & Model Parser<br/>(Land Rover, Mercedes-Benz, Ashok Leyland)"]
        BrandExtractor --> UnitParser["Regex Normalizer & Standardizer<br/>mileage, engine CC, max_power bhp, torque"]
        UnitParser --> Imputer["Deterministic Median Imputation<br/>mileage: 19.3 kmpl | engine: 1248 CC | power: 82 bhp"]
        Imputer --> FeatEng["Automotive Feature Engineering<br/>vehicle_age, km_per_year, power_to_engine_ratio, is_luxury"]
        FeatEng --> ParquetOut[("Optimized Columnar Parquet<br/>data/processed/vehicles.parquet<br/>(6,946 clean records)")]
    end

    subgraph S3 ["3. Analytical & Machine Learning Engines"]
        ParquetOut --> SparkSQL["Apache Spark SQL<br/>Whole-Stage Code Generation<br/>6 Aggregated Market Views"]
        ParquetOut --> GraphX["Scala Spark GraphX<br/>Heterogeneous Bipartite Graph<br/>PageRank & Connected Components"]
        ParquetOut --> MLlib["Spark MLlib Pipeline<br/>StringIndexer(keep) + OneHotEncoder + VectorAssembler<br/>RandomForestRegressor & GBTRegressor"]
    end

    subgraph S4 ["4. Application & Delivery"]
        SparkSQL --> Page1["Page 1: Market Overview & SQL Analytics"]
        GraphX --> Page2["Page 2: Graph Intelligence & Market Hubs"]
        MLlib --> Page3["Page 3: Real-Time Valuation & Inference"]
        ParquetOut --> Page4["Page 4: Empirical Scalability Benchmarks"]
        S4 --> ReportGen["Academic Report Generator<br/>generate_report.py -> DOCX Report"]
    end

    style S1 fill:#f0f7ff,stroke:#0366d6,stroke-width:2px
    style S2 fill:#fef9e7,stroke:#f5b041,stroke-width:2px
    style S3 fill:#fdf2e9,stroke:#e67e22,stroke-width:2px
    style S4 fill:#eafaf1,stroke:#2ecc71,stroke-width:2px
```

---

## 📁 Directory Structure

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
├── legacy/                      # Baseline reference implementation
│   ├── app.py                   # Original single-node script
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
├── CarIntel_INT315_Project_Report.docx # Academic project report document
├── generate_report.py           # Report generator using python-docx
├── pytest.ini                   # Pytest discovery configuration
├── requirements.txt             # Unified Python dependencies
└── LICENSE                      # MIT Open Source License
```

---

## 🔬 Machine Learning Prediction Values & Metrics

Models were evaluated on the clean deduplicated dataset (6,946 records) using a deterministic 80/20 train/test split:

| Model Architecture | Train $R^2$ | Test $R^2$ | Test RMSE | Test MAE | Data Leakage Controlled | Unseen Category Resilience |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Spark MLlib Random Forest** | **0.9620** | **0.9000** | **₹147,816** | **₹81,699** | ✅ Clean (0 duplicates) | ✅ Crash-Proof (`keep`) |
| **Spark MLlib GBT** | 0.9805 | 0.7915 | ₹213,456 | ₹88,211 | ✅ Clean (0 duplicates) | ✅ Crash-Proof (`keep`) |

### Top Predictive Feature Importances
1. **`max_power_bhp`**: Peak engine horsepower output (primary valuation driver)
2. **`engine_cc`**: Total engine displacement in cubic centimeters
3. **`vehicle_age`**: Vehicle age calculated from base reference year (2025)
4. **`km_driven`**: Cumulative odometer usage

---

## ⚡ Execution Commands

### 1. Environment Setup
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

### 2. Launch Interactive Streamlit Dashboard
You can run the web dashboard using either command:

```powershell
# Standard command
streamlit run dashboard/streamlit_app.py

# Alternative command (preferred on Windows environments)
python -m streamlit run dashboard/streamlit_app.py
```

### 3. Run Distributed ETL Pipeline & SQL Analytics
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

### 6. Run Scalability Benchmarks
```powershell
python -m experiments.synthetic_data_generator
python -m experiments.benchmark_runner
```

### 7. Run Test Suite
```powershell
pytest tests/ -v
```

### 8. Generate Academic Project Report (DOCX)
```powershell
python generate_report.py
```

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
