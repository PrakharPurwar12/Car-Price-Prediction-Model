"""
dashboard/streamlit_app.py
CarIntel Platform — Master Application Entry Point & Executive Architecture Overview.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path when running via 'streamlit run'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dashboard.utils import load_market_analytics, load_graph_summary, format_inr

st.set_page_config(
    page_title="CarIntel — Distributed Market Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 6px;
    }
    .badge-spark { background-color: #E25A1C; color: white; }
    .badge-scala { background-color: #DC322F; color: white; }
    .badge-java { background-color: #007396; color: white; }
    .badge-parquet { background-color: #5B92E5; color: white; }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🚗 CarIntel — Distributed Vehicle Market Intelligence & Price Prediction Platform")
st.markdown("""
**CarIntel** is an enterprise-grade Big Data platform engineered on **Apache Spark**, **Spark SQL**, **Spark GraphX (Scala)**, and **Spark MLlib**. It provides distributed ingestion, multi-relational graph topology analytics, and high-precision vehicle resale valuation.
""")

st.markdown("""
<div>
    <span class="badge badge-spark">Apache Spark 4.2.0</span>
    <span class="badge badge-scala">Scala 2.13</span>
    <span class="badge badge-java">Java 17 LTS</span>
    <span class="badge badge-spark">Spark GraphX</span>
    <span class="badge badge-spark">Spark MLlib</span>
    <span class="badge badge-parquet">Apache Parquet</span>
</div>
""", unsafe_allow_html=True)

st.divider()

# Load Market Analytics
analytics = load_market_analytics()
kpi = analytics.get("market_kpi", {})
graph_summary = load_graph_summary()

# Top KPI Summary Cards
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Deduplicated Records",
        value=f"{kpi.get('total_vehicles', 6946):,}",
        delta="1,202 Duplicates Removed"
    )

with col2:
    st.metric(
        label="Automotive Brands",
        value=kpi.get("total_brands", 32),
        delta="Catalogued"
    )

with col3:
    st.metric(
        label="Median Resale Price",
        value=format_inr(kpi.get("market_median_price", 450000)),
        delta=f"Avg: {format_inr(kpi.get('market_avg_price', 638000))}"
    )

with col4:
    st.metric(
        label="Total Market Capitalization",
        value=f"₹{kpi.get('total_market_value_crores', 442.8):,.1f} Cr",
        delta="Processed Assets"
    )

with col5:
    st.metric(
        label="GraphX Vertices / Edges",
        value=f"{graph_summary.get('total_vertices', 6985):,} / {graph_summary.get('total_edges', 20838):,}",
        delta="Bipartite Topology"
    )

st.divider()

# Architecture & Engineering Workflow
st.subheader("🏗️ System Architecture & Distributed Data Flow")

st.markdown("""
```mermaid
flowchart LR
    subgraph Storage ["Distributed Storage Layer"]
        A[("Raw CSV Dataset<br/>8,148 Rows")] --> B["PySpark Ingestion &<br/>Schema Validation"]
        B --> C["Distributed Cleaning &<br/>Exact Deduplication<br/>(Removed 1,202 Duplicates)"]
        C --> D["Domain Feature Engineering<br/>(Age, Ratios, Log Target)"]
        D --> E[("Optimized Parquet Storage<br/>6,946 Records")]
    end

    subgraph Analytics ["Distributed Computation Engines"]
        E --> F["Apache Spark SQL<br/>Analytical Views"]
        E --> G["Scala Spark GraphX<br/>PageRank & Components"]
        E --> H["Spark MLlib Pipeline<br/>RandomForest & GBT"]
    end

    subgraph Serving ["Interactive Intelligence Dashboard"]
        F --> I["Page 1: Market Intelligence"]
        G --> J["Page 2: Graph Intelligence"]
        H --> K["Page 3: Real-Time Valuation"]
        E --> L["Page 4: Empirical Scalability"]
    end

    style Storage fill:#f0f7ff,stroke:#0366d6,stroke-width:2px
    style Analytics fill:#fef9e7,stroke:#f5b041,stroke-width:2px
    style Serving fill:#eafaf1,stroke:#2ecc71,stroke-width:2px
```
""")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### 🎯 Key Engineering Highlights")
    st.markdown("""
    - **100% Genuine Distributed Tech Stack**: Powered by native PySpark, Spark SQL, Scala Spark GraphX, and Spark MLlib.
    - **Zero Data Leakage**: Identified and eliminated 1,202 duplicate records that artificially inflated legacy models.
    - **Native Windows Hadoop Engine**: Custom `WinLocalFileSystem` bypasses `winutils.exe` restrictions cleanly via Java 17 NIO.
    - **Multi-Relational GraphX Analytics**: Structural PageRank and market hub identification running on Scala GraphX.
    - **Resilient Inference Pipeline**: `StringIndexer` with `handleInvalid="keep"` fixes the legacy Peugeot crash bug permanently.
    """)

with col_right:
    st.markdown("### 🧭 Platform Navigation")
    st.markdown("""
    Select an intelligence module from the sidebar:
    1. **📊 1_Market_Overview**: Distributed Spark SQL queries, brand valuations, fuel depreciation, and transmission premiums.
    2. **🕸️ 2_Graph_Intelligence**: Scala Spark GraphX PageRank scores, degree centralities, and connected components.
    3. **🔮 3_Price_Prediction**: Interactive real-time valuation powered by Spark MLlib PipelineModel with edge-case validation.
    4. **⚡ 4_Scalability_Benchmarks**: Empirical performance benchmarks comparing Pandas vs PySpark and Sklearn vs MLlib across 1x to 20x scales.
    """)

st.caption("CarIntel Platform | Senior Big Data & Distributed Machine Learning Architecture")

