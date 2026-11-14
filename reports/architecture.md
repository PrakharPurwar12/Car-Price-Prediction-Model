# CarIntel — Enterprise Distributed System Architecture

## 1. Architectural Philosophy & Overview

**CarIntel** transforms a conventional single-node Python script into an enterprise-grade Big Data and Distributed Machine Learning platform. The platform is designed around the core principles of:

1. **Decoupled Distributed Compute and Storage:** Leveraging Apache Spark for compute scaling and Apache Parquet for columnar, schema-enforced persistence.
2. **Strict Data Hygiene & Zero Contamination:** Systematic identification and elimination of exact duplicate records before partitioning.
3. **Graph Topology Intelligence:** Modeling multi-relational interactions between vehicles, manufacturers, fuel types, and transmissions using Scala Apache Spark GraphX.
4. **Resilient Production Inference:** Fully serialized `PipelineModel` artifacts equipped with `handleInvalid="keep"` to guarantee zero-crash execution across previously unseen automotive categories.

---

## 2. High-Level System Architecture

```mermaid
flowchart TD
    subgraph Ingestion ["1. Distributed Ingestion & Schema Enforcement"]
        RawCSV[("Raw Dataset<br/>Car dekho.csv<br/>(8,148 rows)")] --> SparkIngest["pyspark.sql.DataFrameReader<br/>mode: DROPMALFORMED<br/>Schema: RAW_CAR_SCHEMA"]
        SparkIngest --> DedupCheck["Exact Deduplication<br/>Identified 1,202 Duplicates (14.75%)"]
    end

    subgraph Preprocessing ["2. Distributed Preprocessing & Feature Engineering"]
        DedupCheck --> BrandExtractor["Multi-Word Brand & Model Parser<br/>(Land Rover, Mercedes-Benz, Ashok Leyland)"]
        BrandExtractor --> UnitParser["Regex Normalizer & Standardizer<br/>mileage (kmpl & km/kg) -> mileage_kmpl_std<br/>engine (CC) -> engine_cc<br/>max_power (bhp) -> max_power_bhp"]
        UnitParser --> Imputer["Deterministic Median Imputation<br/>mileage: 19.3 kmpl | engine: 1248 CC | power: 82 bhp"]
        Imputer --> FeatEng["Automotive Feature Engineering<br/>vehicle_age, km_per_year, power_to_engine_ratio, is_luxury"]
        FeatEng --> ParquetOut[("Optimized Columnar Parquet<br/>data/processed/vehicles.parquet<br/>(6,946 clean records)")]
    end

    subgraph AnalyticsEngines ["3. Analytical & Machine Learning Engines"]
        ParquetOut --> SparkSQL["Apache Spark SQL<br/>Whole-Stage Code Generation<br/>Aggregations & Analytical Queries"]
        ParquetOut --> GraphX["Scala Spark GraphX<br/>Multi-Relational Graph Builder<br/>PageRank & Connected Components"]
        ParquetOut --> MLlib["Spark MLlib Pipeline<br/>StringIndexer + OneHotEncoder + VectorAssembler<br/>RandomForestRegressor & GBTRegressor"]
    end

    subgraph Presentation ["4. Multi-Page Interactive Dashboard (Streamlit)"]
        SparkSQL --> Page1["Page 1: Market Overview & SQL Analytics"]
        GraphX --> Page2["Page 2: Graph Intelligence & Market Hubs"]
        MLlib --> Page3["Page 3: Real-Time Valuation & Inference"]
        ParquetOut --> Page4["Page 4: Empirical Scalability Benchmarks"]
    end

    style Ingestion fill:#f0f7ff,stroke:#0366d6,stroke-width:2px
    style Preprocessing fill:#fef9e7,stroke:#f5b041,stroke-width:2px
    style AnalyticsEngines fill:#fdf2e9,stroke:#e67e22,stroke-width:2px
    style Presentation fill:#eafaf1,stroke:#2ecc71,stroke-width:2px
```

---

## 3. Technology Stack & Component Justification

| Layer | Technology | Version | Architectural Justification |
|---|---|---|---|
| **Runtime** | Java LTS | OpenJDK 17.0.12 | Modern LTS JVM runtime with strong encapsulation and enhanced GC performance. |
| **Distributed Engine** | Apache Spark | 4.2.0 | Unified analytics engine with Catalyst optimizer and Tungsten memory manager. |
| **Python API** | PySpark | 4.3.0 | Distributed Python API providing idiomatic DataFrame & MLlib pipeline integration. |
| **Language** | Scala | 2.13.18 | High-performance functional runtime native to Spark GraphX. |
| **Graph Processing** | Spark GraphX | 4.2.0 | Resilient Distributed Graph (RDG) and Pregel API for topological analytics. |
| **Distributed ML** | Spark MLlib | 4.2.0 | Out-of-core distributed model training across multi-node partitions. |
| **Storage** | Apache Parquet | 26.0 (PyArrow) | Columnar format with Snappy compression, dictionary encoding, and partition pruning. |
| **Dashboard** | Streamlit | 1.67.0 | Reactive Python web framework for multi-page interactive analytics. |
| **Visualization** | Plotly | 7.1.0 | Interactive GPU-accelerated client-side data visualizations. |

---

## 4. Windows Hadoop Storage Engine (`win_local_fs.jar`)

### The Challenge
On Windows operating systems, standard Hadoop file output committers attempt to invoke `winutils.exe chmod` to set POSIX permissions. When running in secure environments or Windows 11 with Application Control (AppLocker), third-party binaries trigger `CreateProcess error=4551`.

### The CarIntel Solution
CarIntel implements a native Java replacement filesystem:
```
com.carintel.hadoop.WinLocalFileSystem
```
- Extends Hadoop's `LocalFileSystem` and `RawLocalFileSystem`.
- Intercepts directory creation and file commit operations.
- Uses Java 17 NIO `java.io.File.mkdirs()` and URI status mappings natively.
- Completely eliminates any dependency on `winutils.exe` or external binaries.
- Packaged into `config/win_local_fs.jar` and injected via `spark.hadoop.fs.file.impl`.

---

## 5. Distributed Machine Learning Pipeline Design

```mermaid
flowchart LR
    A["Raw Input Features<br/>(Categorical & Numeric)"] --> B["StringIndexer<br/>handleInvalid='keep'"]
    B --> C["OneHotEncoder<br/>handleInvalid='keep'"]
    C --> D["VectorAssembler<br/>61-Dimensional Feature Vector"]
    D --> E["RandomForestRegressor<br/>100 Trees, Depth 12"]
    E --> F["PipelineModel Serialization<br/>ml/models/rf_model"]

    style A fill:#f9f9f9,stroke:#333
    style D fill:#fef9e7,stroke:#f5b041
    style F fill:#eafaf1,stroke:#2ecc71
```

### Key Engineering Features:
- **Resilience to Unseen Categories:** By specifying `handleInvalid="keep"`, brands unseen during training (e.g. `Peugeot`) are mapped to a reserved category index rather than throwing runtime exceptions (`KeyError`).
- **Energy-Equivalent Fuel Standardization:** Vehicles using CNG/LPG are standardized to petrol-equivalent mileage:
  $$\text{mileage}_{\text{std}} = \text{mileage}_{\text{km/kg}} \times 1.39$$
- **Leakage-Free Validation:** Strict 80/20 deterministic split executed only after deduplication, achieving a genuine test $R^2 = 0.9000$.

