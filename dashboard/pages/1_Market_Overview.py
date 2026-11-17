"""
dashboard/pages/1_Market_Overview.py
Interactive Market Intelligence & Spark SQL Analytical Queries.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
from dashboard.utils import load_market_analytics, load_sample_vehicles, format_inr, render_table

st.set_page_config(page_title="Market Overview | CarIntel", page_icon="📊", layout="wide")

st.title("📊 Distributed Market Intelligence (Spark SQL)")
st.markdown("Aggregated analytical metrics computed via **Apache Spark SQL** over 6,946 verified vehicles.")

analytics = load_market_analytics()
if not analytics:
    st.error("Market analytics data not found. Please run spark_processing/analytical_queries.py.")
    st.stop()

# 1. Brand Intelligence
st.subheader("1. Brand Market Capitalization & Volume Leaderboard")
brand_df = pd.DataFrame(analytics["brand_summary"])

col1, col2 = st.columns([2, 1])

with col1:
    fig_brand = px.bar(
        brand_df.head(15),
        x="brand",
        y="vehicle_count",
        color="avg_price",
        color_continuous_scale="Viridis",
        title="Top 15 Automotive Brands by Listing Volume & Average Resale Price",
        labels={"vehicle_count": "Active Listings", "brand": "Brand", "avg_price": "Avg Price (₹)"},
        text="vehicle_count"
    )
    fig_brand.update_traces(textposition="outside")
    fig_brand.update_layout(height=450)
    st.plotly_chart(fig_brand, use_container_width=True)

with col2:
    st.markdown("#### High-Value / Luxury Brands")
    luxury_df = brand_df[brand_df["is_luxury"] == 1].sort_values("avg_price", ascending=False).copy()
    luxury_disp = luxury_df[["brand", "vehicle_count", "avg_price", "median_price"]].rename(
        columns={
            "brand": "Brand",
            "vehicle_count": "Listings",
            "avg_price": "Avg Price",
            "median_price": "Median Price"
        }
    )
    luxury_disp["Avg Price"] = luxury_disp["Avg Price"].map("₹{:,.0f}".format)
    luxury_disp["Median Price"] = luxury_disp["Median Price"].map("₹{:,.0f}".format)
    render_table(luxury_disp, max_height=380)

st.divider()

# 2. Fuel Type & Depreciation Analysis
st.subheader("2. Fuel Economy & Residual Retention Analysis")
fuel_df = pd.DataFrame(analytics["fuel_summary"])

col3, col4 = st.columns(2)

with col3:
    fig_fuel = px.bar(
        fuel_df,
        x="fuel",
        y="avg_price",
        color="fuel",
        title="Average Resale Price by Fuel Type",
        labels={"avg_price": "Average Price (₹)", "fuel": "Fuel Type"},
        text_auto=".2s"
    )
    fig_fuel.update_layout(height=400)
    st.plotly_chart(fig_fuel, use_container_width=True)

with col4:
    fig_mileage = px.scatter(
        fuel_df,
        x="avg_mileage",
        y="avg_price",
        size="total_count",
        color="fuel",
        hover_name="fuel",
        title="Standardized Mileage (kmpl) vs. Resale Valuation",
        labels={"avg_mileage": "Standardized Fuel Economy (kmpl)", "avg_price": "Average Resale Price (₹)"}
    )
    fig_mileage.update_layout(height=400)
    st.plotly_chart(fig_mileage, use_container_width=True)

st.divider()

# 3. Transmission Premium & Engine Displacement Segmentation
st.subheader("3. Transmission Premium & Engine Displacement Segments")
trans_df = pd.DataFrame(analytics["transmission_summary"])
engine_df = pd.DataFrame(analytics["engine_segment_summary"])

col5, col6 = st.columns(2)

with col5:
    fig_trans = px.pie(
        trans_df,
        names="transmission",
        values="total_count",
        title="Transmission Market Share (Manual vs Automatic)",
        hole=0.45,
        color_discrete_sequence=["#3498db", "#e74c3c"]
    )
    st.plotly_chart(fig_trans, use_container_width=True)
    
    # Transmission premium highlight
    auto_price = trans_df[trans_df["transmission"] == "Automatic"]["avg_price"].values[0]
    man_price = trans_df[trans_df["transmission"] == "Manual"]["avg_price"].values[0]
    premium = auto_price - man_price
    pct = (premium / man_price) * 100
    st.info(f"💡 **Automatic Transmission Valuation Premium:** {format_inr(premium)} (+{pct:.1f}% higher on average).")

with col6:
    fig_engine = px.bar(
        engine_df,
        x="engine_segment",
        y="avg_price",
        color="avg_power_bhp",
        title="Engine Displacement vs. Resale Valuation",
        labels={"engine_segment": "Engine CC Segment", "avg_price": "Average Price (₹)", "avg_power_bhp": "BHP"},
        text_auto=".2s"
    )
    fig_engine.update_layout(height=400, xaxis_tickangle=-25)
    st.plotly_chart(fig_engine, use_container_width=True)

st.divider()

# 4. Interactive Data Explorer
st.subheader("4. Raw Verified Vehicle Dataset Explorer")
sample_df = load_sample_vehicles(max_rows=500)

brand_filter = st.multiselect(
    "Filter by Brand:",
    options=sorted(sample_df["brand"].dropna().unique().tolist()),
    default=["Maruti", "Hyundai", "BMW", "Mercedes-Benz"]
)

filtered = sample_df[sample_df["brand"].isin(brand_filter)].copy() if brand_filter else sample_df.copy()

disp_filtered = filtered[["brand", "model_name", "year", "selling_price", "km_driven", "fuel", "transmission", "mileage_kmpl_std", "engine_cc", "max_power_bhp"]].rename(
    columns={
        "brand": "Brand",
        "model_name": "Model",
        "year": "Year",
        "selling_price": "Price (₹)",
        "km_driven": "KM Driven",
        "fuel": "Fuel",
        "transmission": "Transmission",
        "mileage_kmpl_std": "Mileage (kmpl)",
        "engine_cc": "Engine (CC)",
        "max_power_bhp": "Power (bhp)"
    }
)
disp_filtered["Price (₹)"] = disp_filtered["Price (₹)"].map("₹{:,.0f}".format)
disp_filtered["KM Driven"] = disp_filtered["KM Driven"].map("{:,.0f}".format)

render_table(disp_filtered, max_height=350)

