"""
dashboard/utils.py
Cached Data Loaders, Formatting Utilities, and Safe Table Renderer for CarIntel Dashboard.
Completely bypasses PyArrow DLL restrictions via clean HTML table rendering.
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_CSV = ROOT_DIR / "data" / "processed" / "vehicles_clean.csv"
MARKET_ANALYTICS_JSON = ROOT_DIR / "data" / "processed" / "sql_analytics" / "market_analytics.json"
GRAPH_SUMMARY_JSON = ROOT_DIR / "data" / "graph_metrics" / "graph_summary.json"
GRAPH_NODES_CSV = ROOT_DIR / "data" / "graph_metrics" / "graph_nodes_clean.csv"
BENCHMARK_METRICS_CSV = ROOT_DIR / "experiments" / "results" / "benchmark_metrics.csv"
RF_METRICS_JSON = ROOT_DIR / "ml" / "models" / "rf_metrics.json"
GBT_METRICS_JSON = ROOT_DIR / "ml" / "models" / "gbt_metrics.json"


@st.cache_data
def load_market_analytics():
    if MARKET_ANALYTICS_JSON.exists():
        with open(MARKET_ANALYTICS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_graph_summary():
    if GRAPH_SUMMARY_JSON.exists():
        with open(GRAPH_SUMMARY_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_graph_nodes():
    if GRAPH_NODES_CSV.exists():
        return pd.read_csv(GRAPH_NODES_CSV)
    return pd.DataFrame()


@st.cache_data
def load_benchmark_metrics():
    if BENCHMARK_METRICS_CSV.exists():
        return pd.read_csv(BENCHMARK_METRICS_CSV)
    return pd.DataFrame()


@st.cache_data
def load_ml_metrics():
    rf = {}
    gbt = {}
    if RF_METRICS_JSON.exists():
        with open(RF_METRICS_JSON, "r", encoding="utf-8") as f:
            rf = json.load(f)
    if GBT_METRICS_JSON.exists():
        with open(GBT_METRICS_JSON, "r", encoding="utf-8") as f:
            gbt = json.load(f)
    return rf, gbt


@st.cache_data
def load_sample_vehicles(max_rows=1000):
    if PROCESSED_CSV.exists():
        df = pd.read_csv(PROCESSED_CSV)
        if len(df) > max_rows:
            return df.sample(max_rows, random_state=42)
        return df
    return pd.DataFrame()


def format_inr(val):
    if val is None or pd.isna(val):
        return "₹0"
    val = float(val)
    if val >= 10000000:
        return f"₹{val / 10000000:.2f} Cr"
    elif val >= 100000:
        return f"₹{val / 100000:.2f} Lakh"
    else:
        return f"₹{val:,.0f}"


def render_table(df: pd.DataFrame, max_height: int = 400):
    """
    Renders a styled, responsive HTML table in Streamlit.
    Avoids st.dataframe() and st.table() PyArrow DLL restrictions.
    """
    css = f"""
    <style>
        .custom-tbl-wrapper {{
            width: 100%;
            max-height: {max_height}px;
            overflow: auto;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            margin: 12px 0 20px 0;
            background: #ffffff;
        }}
        .custom-tbl {{
            width: 100%;
            border-collapse: collapse;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 13.5px;
            color: #212529;
        }}
        .custom-tbl th {{
            position: sticky;
            top: 0;
            background-color: #f8f9fa;
            color: #1a1a1a;
            font-weight: 600;
            text-align: left;
            padding: 10px 14px;
            border-bottom: 2px solid #dee2e6;
            z-index: 2;
        }}
        .custom-tbl td {{
            padding: 9px 14px;
            border-bottom: 1px solid #f1f3f5;
        }}
        .custom-tbl tr:hover {{
            background-color: #f8f9fa;
        }}
    </style>
    """
    html_table = df.to_html(classes="custom-tbl", index=False, escape=False)
    st.markdown(f'{css}<div class="custom-tbl-wrapper">{html_table}</div>', unsafe_allow_html=True)
