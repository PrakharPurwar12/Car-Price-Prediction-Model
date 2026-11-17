"""
dashboard/pages/2_Graph_Intelligence.py
Scala Apache Spark GraphX Analytics Explorer.
Visualizes PageRank structural hubs, degree centrality, and connected components.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
from dashboard.utils import load_graph_summary, load_graph_nodes, format_inr, render_table

st.set_page_config(page_title="Graph Intelligence | CarIntel", page_icon="🕸️", layout="wide")

st.title("🕸️ Graph Intelligence (Scala Spark GraphX)")
st.markdown("Multi-relational bipartite graph analytics executed natively using **Apache Spark GraphX**.")

graph_summary = load_graph_summary()
if not graph_summary:
    st.error("Graph summary data not found. Please run scripts/run_graphx.py.")
    st.stop()

# Graph Topology Overview Cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Vertices (V)", f"{graph_summary['total_vertices']:,}")
with c2:
    st.metric("Total Multi-Relational Edges (E)", f"{graph_summary['total_edges']:,}")
with c3:
    st.metric("Connected Components", f"{graph_summary['connected_components']}")
with c4:
    st.metric("Top PageRank Entity", graph_summary["top_pagerank_nodes"][0]["name"])

st.divider()

# 1. PageRank Prestige Leaderboard
st.subheader("1. PageRank Influence & Market Hub Ranking")
st.markdown("""
PageRank measures the structural centrality and liquidity influence of each node in the automotive market graph.
Nodes with high PageRank act as **structural anchors** connecting diverse vehicle listings.
""")

pr_data = pd.DataFrame(graph_summary["top_pagerank_nodes"])

col1, col2 = st.columns([2, 1])

with col1:
    fig_pr = px.bar(
        pr_data,
        x="name",
        y="pageRank",
        color="nodeType",
        title="Top 20 Market Hubs by GraphX PageRank Score",
        labels={"pageRank": "PageRank Score", "name": "Entity Name", "nodeType": "Node Type"},
        text_auto=".1f"
    )
    fig_pr.update_layout(xaxis_tickangle=-35, height=450)
    st.plotly_chart(fig_pr, use_container_width=True)

with col2:
    st.markdown("#### PageRank Summary Table")
    pr_disp = pr_data[["name", "nodeType", "pageRank", "totalDegree"]].rename(
        columns={
            "name": "Entity",
            "nodeType": "Type",
            "pageRank": "PageRank",
            "totalDegree": "Degree (Degree Centrality)"
        }
    ).copy()
    pr_disp["PageRank"] = pr_disp["PageRank"].map("{:.2f}".format)
    pr_disp["Degree (Degree Centrality)"] = pr_disp["Degree (Degree Centrality)"].map("{:,}".format)
    render_table(pr_disp, max_height=380)

st.divider()

# 2. Node Centrality & Category Breakdown
st.subheader("2. Node Type Distribution & Degree Centrality")
nodes_df = load_graph_nodes()

if not nodes_df.empty:
    col3, col4 = st.columns(2)

    with col3:
        type_counts = nodes_df["nodeType"].value_counts().reset_index()
        type_counts.columns = ["nodeType", "count"]
        fig_types = px.pie(
            type_counts,
            names="nodeType",
            values="count",
            title="Graph Vertex Composition by Entity Type",
            hole=0.4
        )
        st.plotly_chart(fig_types, use_container_width=True)

    with col4:
        brand_nodes = nodes_df[nodes_df["nodeType"] == "BRAND"].sort_values("totalDegree", ascending=False)
        fig_deg = px.bar(
            brand_nodes.head(10),
            x="name",
            y="totalDegree",
            title="Top 10 Brand Hubs by Degree Centrality",
            labels={"totalDegree": "Total Incident Edges", "name": "Brand"},
            color="totalDegree",
            color_continuous_scale="Teal"
        )
        fig_deg.update_layout(height=400)
        st.plotly_chart(fig_deg, use_container_width=True)
else:
    st.info("Detailed node parquet dataset not cached.")

st.caption("Scala Spark GraphX Engine | Connected Components & Structural PageRank Algorithms")

