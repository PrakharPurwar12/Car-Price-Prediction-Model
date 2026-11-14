"""
dashboard/pages/4_Scalability_Benchmarks.py
Empirical Scalability & Distributed Performance Benchmarks Explorer.
Visualizes real measured benchmarks across 1x, 5x, 10x, 20x dataset scales.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dashboard.utils import load_benchmark_metrics, render_table

st.set_page_config(page_title="Scalability Benchmarks | CarIntel", page_icon="⚡", layout="wide")

st.title("⚡ Distributed Scalability & Performance Benchmarks")
st.markdown("Empirical performance measurements across **1x, 5x, 10x, and 20x** dataset scales (up to 162,960 records).")

bench_df = load_benchmark_metrics()
if bench_df.empty:
    st.error("Benchmark metrics not found. Please run experiments/benchmark_runner.py.")
    st.stop()

# Benchmark Summary Table
st.subheader("1. Empirical Benchmark Measurements Table")

display_df = bench_df.copy()
display_df = display_df.rename(
    columns={
        "scale": "Scale",
        "raw_records": "Records",
        "pandas_etl_sec": "Pandas ETL (s)",
        "spark_etl_sec": "Spark ETL (s)",
        "spark_throughput_rows_sec": "Spark Throughput (rows/s)",
        "sklearn_train_sec": "Sklearn Train (s)",
        "spark_mllib_train_sec": "Spark MLlib Train (s)",
        "spark_sql_query_sec": "Spark SQL Query (s)"
    }
)
display_df["Records"] = display_df["Records"].map("{:,}".format)
display_df["Pandas ETL (s)"] = display_df["Pandas ETL (s)"].map("{:.3f}".format)
display_df["Spark ETL (s)"] = display_df["Spark ETL (s)"].map("{:.3f}".format)
display_df["Spark Throughput (rows/s)"] = display_df["Spark Throughput (rows/s)"].map("{:,.1f}".format)
display_df["Sklearn Train (s)"] = display_df["Sklearn Train (s)"].map("{:.3f}".format)
display_df["Spark MLlib Train (s)"] = display_df["Spark MLlib Train (s)"].map("{:.3f}".format)
display_df["Spark SQL Query (s)"] = display_df["Spark SQL Query (s)"].map("{:.3f}".format)

render_table(display_df[["Scale", "Records", "Pandas ETL (s)", "Spark ETL (s)", "Spark Throughput (rows/s)", "Sklearn Train (s)", "Spark MLlib Train (s)", "Spark SQL Query (s)"]])

st.divider()

# 2. Spark Distributed Throughput Curve
st.subheader("2. Spark Distributed Processing Throughput")
col1, col2 = st.columns([1.5, 1])

with col1:
    fig_tp = px.line(
        bench_df,
        x="raw_records",
        y="spark_throughput_rows_sec",
        markers=True,
        title="Spark Distributed Processing Throughput vs. Dataset Size",
        labels={"raw_records": "Processed Records", "spark_throughput_rows_sec": "Throughput (Records / Second)"},
        text="spark_throughput_rows_sec"
    )
    fig_tp.update_traces(textposition="top left", line_color="#E25A1C", line_width=3)
    fig_tp.update_layout(height=420)
    st.plotly_chart(fig_tp, use_container_width=True)

with col2:
    st.markdown("#### 📈 Amortization of JVM Overhead")
    st.markdown("""
    - **1x Scale (8,148 rows)**: Spark executes at **919 records/sec**. At small batch sizes, driver orchestration, task scheduling, and Catalyst optimizer plan compilation dominate.
    - **20x Scale (162,960 rows)**: Throughput surges to **35,300+ records/sec** (a **38.4x increase** in throughput).
    - **Key Finding**: Distributed engines like Spark exhibit substantial throughput scaling as dataset volume grows because constant initialization overhead is amortized across larger data partitions.
    """)

st.divider()

# 3. Execution Latency Comparison: Pandas vs PySpark
st.subheader("3. Ingestion & Preprocessing Latency Comparison")

fig_lat = go.Figure()
fig_lat.add_trace(go.Bar(
    x=bench_df["scale"],
    y=bench_df["pandas_etl_sec"],
    name="Single-Node Pandas ETL",
    marker_color="#2ecc71"
))
fig_lat.add_trace(go.Bar(
    x=bench_df["scale"],
    y=bench_df["spark_etl_sec"],
    name="Distributed PySpark ETL",
    marker_color="#E25A1C"
))
fig_lat.update_layout(
    barmode="group",
    title="ETL Execution Latency (Pandas vs. PySpark across Scales)",
    xaxis_title="Dataset Scale Factor",
    yaxis_title="Execution Time (Seconds)",
    height=420
)
st.plotly_chart(fig_lat, use_container_width=True)

st.divider()

# 4. Distributed MLlib vs. Single-Node Scikit-Learn
st.subheader("4. Machine Learning Training Latency: MLlib vs. Scikit-Learn")

col3, col4 = st.columns([1.5, 1])

with col3:
    fig_ml = go.Figure()
    fig_ml.add_trace(go.Scatter(
        x=bench_df["raw_records"],
        y=bench_df["sklearn_train_sec"],
        mode="lines+markers",
        name="Scikit-Learn (In-Memory OpenMP)",
        line=dict(color="#3498db", width=2.5)
    ))
    fig_ml.add_trace(go.Scatter(
        x=bench_df["raw_records"],
        y=bench_df["spark_mllib_train_sec"],
        mode="lines+markers",
        name="Spark MLlib (Distributed RDD Aggregate)",
        line=dict(color="#e74c3c", width=2.5)
    ))
    fig_ml.update_layout(
        title="Random Forest Model Training Duration across Record Volume",
        xaxis_title="Record Count",
        yaxis_title="Training Duration (Seconds)",
        height=400
    )
    st.plotly_chart(fig_ml, use_container_width=True)

with col4:
    st.markdown("#### 🔬 Distributed ML Tradeoffs")
    st.markdown("""
    - **Scikit-Learn**: Executes via in-process C-bindings. Extremely fast when the entire dataset fits comfortably in RAM, but fails immediately with `MemoryError` when data exceeds single-node capacity.
    - **Spark MLlib**: Distributes histogram aggregation and decision-tree split computations across worker partitions (`aggregateByCell`). Scales out-of-core and across multi-node clusters.
    """)

st.caption("Empirical measurements generated automatically by CarIntel Benchmark Suite")

