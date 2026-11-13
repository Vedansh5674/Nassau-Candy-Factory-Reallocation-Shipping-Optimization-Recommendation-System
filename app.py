"""
Factory Reallocation & Shipping Optimization Recommendation System
Executive Dashboard & Factory Optimization Simulator for Nassau Candy Distributor.

Final-Year B.Tech Computer Science Project.
Built with Streamlit, Pandas, Plotly, NumPy, and Scikit-learn.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
from typing import Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import data loading and validation module from src/data_loader.py
from src.data_loader import load_data, validate_dataset
# Import simulator module from src/simulator.py
from src.simulator import render_simulator_page
# Import geographic network module from src/geo_network.py
from src.geo_network import render_geographic_network_page
# Import business impact module from src/business_impact.py
from src.business_impact import render_business_impact_page

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Nassau Candy Factory Optimization System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for KPI Cards, Badges, and Professional Executive Layout
st.markdown(
    """
    <style>
    :root {
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --text-muted: #64748B;
        --card-bg: #FFFFFF;
        --card-border: #E2E8F0;
        --card-hover-border: #CBD5E1;
        --box-bg: #F8FAFC;
        --box-text: #334155;
        --scope-bg: #F1F5F9;
        --scope-border: #0284C7;
        --scope-text: #0F172A;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --text-primary: #F8FAFC;
            --text-secondary: #CBD5E1;
            --text-muted: #94A3B8;
            --card-bg: #1E293B;
            --card-border: #334155;
            --card-hover-border: #475569;
            --box-bg: #0F172A;
            --box-text: #E2E8F0;
            --scope-bg: #1E293B;
            --scope-border: #38BDF8;
            --scope-text: #F8FAFC;
        }
    }

    /* Support Streamlit native dark theme */
    [data-theme="dark"], .stApp[class*="theme--dark"], [data-testid="stAppViewContainer"].dark {
        --text-primary: #F8FAFC;
        --text-secondary: #CBD5E1;
        --text-muted: #94A3B8;
        --card-bg: #1E293B;
        --card-border: #334155;
        --card-hover-border: #475569;
        --box-bg: #0F172A;
        --box-text: #E2E8F0;
        --scope-bg: #1E293B;
        --scope-border: #38BDF8;
        --scope-text: #F8FAFC;
    }

    .main-title {
        font-size: 2.15rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .sub-title {
        font-size: 1.05rem;
        color: var(--text-secondary);
        margin-bottom: 1.25rem;
    }
    .kpi-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 14px;
        min-height: 112px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.12);
        border-color: var(--card-hover-border);
    }
    .kpi-title {
        font-size: 0.80rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.2;
    }
    .kpi-subtext {
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-top: 5px;
    }
    .disclaimer-box {
        background-color: var(--box-bg);
        border-left: 4px solid #3B82F6;
        padding: 16px 20px;
        border-radius: 6px;
        margin-top: 1.75rem;
        margin-bottom: 1.5rem;
        color: var(--box-text);
        font-size: 0.90rem;
        line-height: 1.55;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# STEP 1: Data Loading & Validation Pipeline
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_validated_data() -> pd.DataFrame:
    """
    Loads and validates the Nassau Candy dataset using src.data_loader.
    Stops dashboard rendering if validation fails.
    """
    try:
        df = load_data(validate=True, parse_dates=True)
        # Explicit validation check to ensure contract compliance
        validation_report = validate_dataset(df)
        if not validation_report["is_valid"]:
            error_details = "; ".join(
                validation_report.get("errors", ["Dataset schema verification failed."])
            )
            st.error(f"❌ Dataset Validation Failed: {error_details}")
            st.stop()
        return df
    except Exception as e:
        st.error(f"❌ Failed to load and validate dataset: {str(e)}")
        st.stop()


def render_executive_dashboard(df_raw: pd.DataFrame):
    """
    Renders the executive dashboard page: KPI cards, interactive Plotly charts,
    sidebar filters, summary recommendation table, and business limitations.
    """
    # ---------------------------------------------------------
    # Sidebar - Interactive Filters
    # ---------------------------------------------------------
    st.sidebar.title("Dashboard Filters")
    st.sidebar.markdown(
        "Filter recommendations across all KPI cards, visualizations, and summary tables."
    )

    # Extract options from dataset
    factories = sorted(df_raw["Current_Factory"].dropna().unique().tolist())
    products = sorted(df_raw["Product Name"].dropna().unique().tolist())
    decisions = sorted(df_raw["Final_Optimization_Decision"].dropna().unique().tolist())
    risks = sorted(df_raw["Optimization_Risk"].dropna().unique().tolist())

    # Initialize session state for filter persistence and clean resets
    if "filter_factory" not in st.session_state:
        st.session_state["filter_factory"] = factories
    if "filter_product" not in st.session_state:
        st.session_state["filter_product"] = products
    if "filter_decision" not in st.session_state:
        st.session_state["filter_decision"] = decisions
    if "filter_risk" not in st.session_state:
        st.session_state["filter_risk"] = risks

    def reset_filters_callback():
        """Resets all sidebar filter selections back to default (all selected)."""
        st.session_state["filter_factory"] = factories
        st.session_state["filter_product"] = products
        st.session_state["filter_decision"] = decisions
        st.session_state["filter_risk"] = risks

    # 1. Factory Filter
    selected_factories = st.sidebar.multiselect(
        "Current Factory",
        options=factories,
        key="filter_factory",
        help="Filter by the currently assigned production facility.",
    )

    # 2. Product Filter
    selected_products = st.sidebar.multiselect(
        "Product Name",
        options=products,
        key="filter_product",
        help="Filter by specific candy products.",
    )

    # 3. Final Decision Filter
    selected_decisions = st.sidebar.multiselect(
        "Final Optimization Decision",
        options=decisions,
        key="filter_decision",
        help="Filter by reallocation recommendation status (RELOCATE - HIGH, RELOCATE - REVIEW, KEEP CURRENT).",
    )

    # 4. Optimization Risk Filter
    selected_risks = st.sidebar.multiselect(
        "Optimization Risk",
        options=risks,
        key="filter_risk",
        help="Filter by operational risk classification (HIGH RISK, MEDIUM RISK, NO ACTION).",
    )

    # Reset filters button
    st.sidebar.button(
        "🔄 Reset All Filters",
        on_click=reset_filters_callback,
        use_container_width=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        **Dataset Status:**  
        • Records: `{len(df_raw):,}`  
        • Columns: `{len(df_raw.columns)}`  
        • Schema: `✅ Validated`  
        
        🎓 **B.Tech Capstone Project**  
        🏭 **Partner**: Nassau Candy Distributor
        """
    )

    # Check if any filter is completely empty
    if not selected_factories or not selected_products or not selected_decisions or not selected_risks:
        st.warning("⚠️ At least one filter has no options selected. Please select options or click 'Reset All Filters'.")
        return

    # Apply Filters consistently
    mask = (
        df_raw["Current_Factory"].isin(selected_factories)
        & df_raw["Product Name"].isin(selected_products)
        & df_raw["Final_Optimization_Decision"].isin(selected_decisions)
        & df_raw["Optimization_Risk"].isin(selected_risks)
    )
    filtered_df = df_raw[mask].copy()

    # ---------------------------------------------------------
    # Header & Context
    # ---------------------------------------------------------
    st.markdown(
        '<div class="main-title">🏭 Factory Reallocation & Shipping Optimization</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Executive Decision-Support Dashboard | Nassau Candy Distributor</div>',
        unsafe_allow_html=True,
    )

    # Check for Empty Filtered Selection
    if filtered_df.empty:
        st.warning("⚠️ No orders match the selected filter criteria. Please broaden your sidebar filter selections.")
        return

    # ---------------------------------------------------------
    # STEP 2: Executive KPI Cards (Calculated Dynamically)
    # ---------------------------------------------------------
    total_orders = len(filtered_df)
    total_sales = float(filtered_df["Sales"].sum())
    total_profit = float(filtered_df["Gross Profit"].sum())
    total_units = int(filtered_df["Units"].sum())

    realloc_candidates_mask = filtered_df["Final_Optimization_Decision"].isin(
        ["RELOCATE - HIGH", "RELOCATE - REVIEW"]
    )
    realloc_candidates = int(realloc_candidates_mask.sum())
    candidate_df = filtered_df[realloc_candidates_mask]

    high_priority_realloc = int(
        (filtered_df["Final_Optimization_Decision"] == "RELOCATE - HIGH").sum()
    )

    total_distance_saved_km = float(filtered_df["Alternative_Distance_Saving_KM"].sum())
    candidate_distance_saved_km = (
        float(candidate_df["Alternative_Distance_Saving_KM"].sum())
        if not candidate_df.empty
        else 0.0
    )

    avg_distance_saving_pct = float(filtered_df["Alternative_Distance_Saving_Pct"].mean())
    candidate_distance_saving_pct = (
        float(candidate_df["Alternative_Distance_Saving_Pct"].mean())
        if not candidate_df.empty
        else 0.0
    )

    total_current_distance_km = float(filtered_df["Current_Factory_Distance_KM"].sum())
    aggregate_distance_saving_pct = (
        float((total_distance_saved_km / total_current_distance_km) * 100.0)
        if total_current_distance_km > 0
        else 0.0
    )

    # Row 1: Primary Order & Financial KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Orders</div>
                <div class="kpi-value">{total_orders:,}</div>
                <div class="kpi-subtext">{(total_orders / len(df_raw) * 100):.1f}% of total dataset ({len(df_raw):,})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Sales</div>
                <div class="kpi-value">${total_sales:,.2f}</div>
                <div class="kpi-subtext">Avg: ${(total_sales / max(1, total_orders)):.2f} / order</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Gross Profit</div>
                <div class="kpi-value">${total_profit:,.2f}</div>
                <div class="kpi-subtext">Margin: {((total_profit / max(1.0, total_sales)) * 100):.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Units</div>
                <div class="kpi-value">{total_units:,}</div>
                <div class="kpi-subtext">Avg: {(total_units / max(1, total_orders)):.1f} units / order</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Row 2: Reallocation Candidates & Potential Geographic Savings
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Reallocation Candidates</div>
                <div class="kpi-value">{realloc_candidates:,}</div>
                <div class="kpi-subtext">{((realloc_candidates / max(1, total_orders)) * 100):.1f}% of active filtered orders</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col6:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">High-Priority Recommendations</div>
                <div class="kpi-value">{high_priority_realloc:,}</div>
                <div class="kpi-subtext">{((high_priority_realloc / max(1, total_orders)) * 100):.1f}% immediate action priority</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col7:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Potential Geographic Distance Savings</div>
                <div class="kpi-value">{total_distance_saved_km:,.0f} km</div>
                <div class="kpi-subtext">Candidates only: {candidate_distance_saved_km:,.0f} km</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col8:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Average Potential Distance Saving Percentage</div>
                <div class="kpi-value">{avg_distance_saving_pct:.1f}%</div>
                <div class="kpi-subtext">Order-level mean (Candidates: {candidate_distance_saving_pct:.1f}% | Net Aggregate: {aggregate_distance_saving_pct:+.1f}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "💡 **Distance Metric Interpretation:** 'Average Potential Distance Saving Percentage' (-18.1%) reflects the unweighted "
        "arithmetic mean of per-order saving percentages across all orders. Non-relocated orders retain their nearby current factory, "
        "so alternative facilities would add substantial geographic distance (negative saving = additional distance). "
        f"For flagged Reallocation Candidates, the mean saving is **+{candidate_distance_saving_pct:.1f}%**, and the network-wide net "
        f"aggregate distance reduction is **+{aggregate_distance_saving_pct:.1f}%** ({total_distance_saved_km:,.0f} km saved)."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # STEP 3: Interactive Plotly Charts
    # ---------------------------------------------------------
    st.markdown('<div class="section-header">📊 Strategic Decision & Operational Risk Distributions</div>', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    # Chart 1: Distribution of Final Optimization Decisions
    with chart_col1:
        st.subheader("1. Optimization Decision Breakdown")
        dec_counts = filtered_df["Final_Optimization_Decision"].value_counts().reset_index()
        dec_counts.columns = ["Decision", "Count"]
        dec_color_map = {
            "KEEP CURRENT": "#64748B",
            "RELOCATE - REVIEW": "#F59E0B",
            "RELOCATE - HIGH": "#10B981",
        }
        fig_dec = px.pie(
            dec_counts,
            names="Decision",
            values="Count",
            color="Decision",
            color_discrete_map=dec_color_map,
            hole=0.45,
            title="Distribution of Final Optimization Decisions",
        )
        fig_dec.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Orders: %{value:,}<br>Proportion: %{percent}<extra></extra>",
        )
        fig_dec.update_layout(
            margin=dict(t=50, b=20, l=20, r=20),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            height=370,
        )
        st.plotly_chart(fig_dec, use_container_width=True)

    # Chart 2: Distribution of Optimization Risk Categories
    with chart_col2:
        st.subheader("2. Operational Risk Classification")
        risk_counts = filtered_df["Optimization_Risk"].value_counts().reset_index()
        risk_counts.columns = ["Risk_Level", "Count"]
        risk_color_map = {
            "NO ACTION": "#94A3B8",
            "MEDIUM RISK": "#FBBF24",
            "HIGH RISK": "#EF4444",
        }
        fig_risk = px.bar(
            risk_counts,
            x="Risk_Level",
            y="Count",
            color="Risk_Level",
            color_discrete_map=risk_color_map,
            title="Orders by Operational Optimization Risk Level",
            labels={"Risk_Level": "Risk Category", "Count": "Order Volume"},
            text="Count",
        )
        fig_risk.update_traces(
            texttemplate="%{text:,}",
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Orders: %{y:,}<extra></extra>",
        )
        fig_risk.update_layout(
            margin=dict(t=50, b=20, l=20, r=20),
            showlegend=False,
            height=370,
            yaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown('<div class="section-header">🚚 Factory Distance Dynamics & Product Saving Potentials</div>', unsafe_allow_html=True)

    chart_col3, chart_col4 = st.columns(2)

    # Chart 3: Average Current vs Alternative Geographic Distance by Current Factory
    with chart_col3:
        st.subheader("3. Current vs. Alternative Distance by Factory")
        dist_factory = (
            filtered_df.groupby("Current_Factory")[
                ["Current_Factory_Distance_KM", "Best_Alternative_Distance_KM"]
            ]
            .mean()
            .reset_index()
        )
        fig_dist = go.Figure()
        fig_dist.add_trace(
            go.Bar(
                name="Current Factory Distance",
                x=dist_factory["Current_Factory"],
                y=dist_factory["Current_Factory_Distance_KM"],
                marker_color="#64748B",
                hovertemplate="%{x}<br>Current Avg: %{y:,.1f} km<extra></extra>",
            )
        )
        fig_dist.add_trace(
            go.Bar(
                name="Best Alternative Distance",
                x=dist_factory["Current_Factory"],
                y=dist_factory["Best_Alternative_Distance_KM"],
                marker_color="#3B82F6",
                hovertemplate="%{x}<br>Alternative Avg: %{y:,.1f} km<extra></extra>",
            )
        )
        fig_dist.update_layout(
            barmode="group",
            title="Avg Current vs. Best Alternative Distance (KM)",
            xaxis_title="Current Manufacturing Facility",
            yaxis_title="Geographic Distance (KM)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=50, b=20, l=20, r=20),
            height=400,
            yaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # Chart 4: Top 10 Products by Potential Geographic Distance Savings
    with chart_col4:
        st.subheader("4. Top 10 Products by Potential Distance Savings")
        top_products = (
            filtered_df.groupby("Product Name")["Alternative_Distance_Saving_KM"]
            .sum()
            .reset_index()
            .sort_values(by="Alternative_Distance_Saving_KM", ascending=True)
            .tail(10)
        )
        fig_prod = px.bar(
            top_products,
            x="Alternative_Distance_Saving_KM",
            y="Product Name",
            orientation="h",
            color="Alternative_Distance_Saving_KM",
            color_continuous_scale="Blues",
            title="Top 10 Products by Total Potential Savings (KM)",
            labels={
                "Alternative_Distance_Saving_KM": "Potential Savings (KM)",
                "Product Name": "Confectionery Product",
            },
        )
        fig_prod.update_traces(
            hovertemplate="<b>%{y}</b><br>Potential Savings: %{x:,.0f} km<extra></extra>",
        )
        fig_prod.update_layout(
            margin=dict(t=50, b=20, l=20, r=20),
            height=400,
            coloraxis_showscale=False,
            xaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
        )
        st.plotly_chart(fig_prod, use_container_width=True)

    # Chart 5: Current Factory vs Recommended Alternative Factory Flow
    st.subheader("5. Factory Reallocation Reassignment Distribution")
    flow_df = (
        filtered_df.groupby(["Current_Factory", "Best_Alternative_Factory"])
        .size()
        .reset_index(name="Order_Count")
    )
    fig_flow = px.bar(
        flow_df,
        x="Current_Factory",
        y="Order_Count",
        color="Best_Alternative_Factory",
        title="Current Factory vs. Recommended Alternative Factory Assignment Flow",
        labels={
            "Current_Factory": "Current Assigned Facility",
            "Order_Count": "Number of Orders",
            "Best_Alternative_Factory": "Recommended Alternative Facility",
        },
        barmode="stack",
        color_discrete_sequence=px.colors.qualitative.Prism,
    )
    fig_flow.update_traces(
        hovertemplate="<b>Current:</b> %{x}<br><b>Recommended:</b> %{fullData.name}<br><b>Orders:</b> %{y:,}<extra></extra>",
    )
    fig_flow.update_layout(
        margin=dict(t=50, b=20, l=20, r=20),
        height=420,
        legend=dict(title="Alternative Factory", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(showgrid=True, gridcolor="#E2E8F0"),
    )
    st.plotly_chart(fig_flow, use_container_width=True)

    # ---------------------------------------------------------
    # STEP 5: High-Priority Recommendation Table & CSV Export
    # ---------------------------------------------------------
    st.markdown('<div class="section-header">📋 Prioritized Fulfillment Recommendations</div>', unsafe_allow_html=True)
    st.markdown(
        "Detailed order-level recommendations ranked by **Recommendation Score** (multi-criteria heuristic score combining "
        "distance reduction percentage, volume burden reduction, product profit margin, and order unit volume)."
    )

    required_table_cols = [
        "Order ID",
        "City",
        "Product Name",
        "Current_Factory",
        "Best_Alternative_Factory",
        "Current_Factory_Distance_KM",
        "Best_Alternative_Distance_KM",
        "Alternative_Distance_Saving_KM",
        "Alternative_Distance_Saving_Pct",
        "Recommendation_Score",
        "Final_Optimization_Decision",
        "Optimization_Risk",
    ]

    table_df = filtered_df[required_table_cols].sort_values(
        by="Recommendation_Score", ascending=False
    )

    display_df = table_df.head(200).copy()

    st.dataframe(
        display_df.style.format(
            {
                "Current_Factory_Distance_KM": "{:,.1f} km",
                "Best_Alternative_Distance_KM": "{:,.1f} km",
                "Alternative_Distance_Saving_KM": "{:,.1f} km",
                "Alternative_Distance_Saving_Pct": "{:.1f}%",
                "Recommendation_Score": "{:.2f}",
            }
        ),
        use_container_width=True,
        height=380,
    )

    # Download Button (always exports all filtered records)
    csv_data = table_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download Filtered Recommendations ({len(table_df):,} Records as CSV)",
        data=csv_data,
        file_name="nassau_candy_recommendations_filtered.csv",
        mime="text/csv",
        help="Export the entire filtered recommendation dataset as a CSV spreadsheet.",
    )

    # ---------------------------------------------------------
    # STEP 6: Data Quality and Business Limitations
    # ---------------------------------------------------------
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Data Quality, Business Modeling & Operational Limitations:</strong><br>
            • <strong>Geographic Approximation:</strong> Factory and customer destination distances are computed using the 
              spherical Haversine formula based on coordinate approximations. They represent potential geographic distance 
              improvements and do not model actual road transit topography, highway routes, or commercial carrier waypoints.<br>
            • <strong>Potential vs. Guaranteed Savings:</strong> Distance savings are reported strictly as potential geographic 
              improvements. They do not constitute guaranteed commercial freight-cost reductions or guaranteed lead-time reductions, 
              which depend on freight carrier contracts, lane rates, volume commitments, and logistics network congestion.<br>
            • <strong>Decision Support Framework:</strong> The recommendation scores (0–100) and risk classifications 
              (High Risk, Medium Risk, No Action) are heuristic decision-support measures designed to prioritize orders for 
              managerial review and scenario planning, not automated dispatch rules.<br>
            • <strong>Operational Feasibility:</strong> Final production and fulfillment reallocation must take into account 
              factory machinery capabilities, confectionery production line constraints, shift scheduling, and warehouse storage.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Application Entry Point & Navigation
# ---------------------------------------------------------
# Load validated dataset
df_raw = get_validated_data()

# Sidebar: Navigation Mode
st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 0.85rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--card-border, #E2E8F0);">
        <span style="font-size: 2.2rem; line-height: 1;">🏭</span>
        <div>
            <div style="font-weight: 700; font-size: 1.15rem; line-height: 1.2; color: var(--text-primary, #0F172A);">Nassau Candy</div>
            <div style="font-size: 0.78rem; color: var(--text-muted, #64748B);">Distribution Optimization</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.title("Navigation")
nav_page = st.sidebar.radio(
    "Select View",
    [
        "Executive Dashboard",
        "Factory Optimization Simulator",
        "Geographic Network",
        "Business Impact & Decision Intelligence",
    ],
    index=0,
    key="nav_page",
    help="Switch between Executive Dashboard, Factory Optimization Simulator, Geographic Network, and Business Impact & Decision Intelligence.",
)
st.sidebar.markdown("---")

if nav_page == "Executive Dashboard":
    render_executive_dashboard(df_raw)
elif nav_page == "Factory Optimization Simulator":
    render_simulator_page(df_raw)
elif nav_page == "Geographic Network":
    render_geographic_network_page(df_raw)
elif nav_page == "Business Impact & Decision Intelligence":
    render_business_impact_page(df_raw)
