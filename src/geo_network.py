"""
Factory Reallocation & Shipping Optimization Recommendation System
Geographic Network & Factory Location Analysis Module for Nassau Candy Distributor.

Provides geographic visualizations, spatial network analysis, factory location maps,
customer centroid distribution, what-if reallocation comparisons with straight-line
geographic connections, and regional concentration insights.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Reuse validated metadata, tolerance, and simulation engine from src.simulator
from src.simulator import (
    FACTORY_METADATA,
    ALL_FACTORIES,
    DISTANCE_TOLERANCE_KM,
    get_factory_distance,
    run_factory_simulation,
    validate_simulation_inputs,
)

# Established distinctive color mapping for the 5 manufacturing facilities
FACTORY_COLORS: Dict[str, str] = {
    "Lot's O' Nuts": "#2563EB",       # Royal Blue
    "Wicked Choccy's": "#059669",     # Emerald Green
    "Sugar Shack": "#D97706",         # Amber Orange
    "Secret Factory": "#7C3AED",      # Purple
    "The Other Factory": "#E11D48",   # Crimson Rose
}

# Decision color mapping
DECISION_COLORS: Dict[str, str] = {
    "KEEP CURRENT": "#3B82F6",        # Blue
    "RELOCATE - REVIEW": "#F59E0B",   # Amber
    "RELOCATE - HIGH": "#10B981",     # Emerald
}

# Comparison impact colors
IMPACT_COLORS: Dict[str, str] = {
    "Decreased Distance (Shorter)": "#10B981",  # Green
    "Increased Distance (Longer)": "#EF4444",   # Red
    "Unchanged Distance": "#94A3B8",            # Slate Gray
}


def get_factory_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes summary metrics for all five manufacturing facilities,
    including coordinates, division, assigned orders, total units, and sales.
    """
    rows = []
    total_orders = len(df)
    for name, meta in FACTORY_METADATA.items():
        f_df = df[df["Current_Factory"] == name]
        orders = len(f_df)
        units = int(f_df["Units"].sum()) if not f_df.empty else 0
        sales = float(f_df["Sales"].sum()) if not f_df.empty else 0.0
        pct_share = (orders / total_orders * 100.0) if total_orders > 0 else 0.0

        rows.append({
            "Factory": name,
            "City": meta["city"],
            "State": meta["state"],
            "Latitude": meta["latitude"],
            "Longitude": meta["longitude"],
            "Specialization": meta["division"],
            "Assigned Orders": orders,
            "Network Share (%)": round(pct_share, 2),
            "Total Units": units,
            "Total Sales ($)": round(sales, 2),
        })

    return pd.DataFrame(rows)


def validate_geo_coordinates(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates the customer coordinates in the dataset to verify geographic integrity.
    Verifies that customer locations are not collapsed into a single centroid.
    """
    results: Dict[str, Any] = {
        "total_rows": len(df),
        "lat_nulls": int(df["Latitude"].isnull().sum()) if "Latitude" in df.columns else -1,
        "lon_nulls": int(df["Longitude"].isnull().sum()) if "Longitude" in df.columns else -1,
        "unique_locations": int(df[["Latitude", "Longitude"]].drop_duplicates().shape[0])
        if "Latitude" in df.columns and "Longitude" in df.columns else 0,
        "lat_min": float(df["Latitude"].min()) if "Latitude" in df.columns else 0.0,
        "lat_max": float(df["Latitude"].max()) if "Latitude" in df.columns else 0.0,
        "lon_min": float(df["Longitude"].min()) if "Longitude" in df.columns else 0.0,
        "lon_max": float(df["Longitude"].max()) if "Longitude" in df.columns else 0.0,
        "has_single_centroid_error": False,
    }

    # Flag critical error if all coordinates are identical
    if results["unique_locations"] <= 1 and results["total_rows"] > 1:
        results["has_single_centroid_error"] = True

    return results


def build_factory_location_map(df: pd.DataFrame) -> go.Figure:
    """
    Generates a Plotly map displaying all five manufacturing facilities
    with their validated coordinates, specialization, and assigned volume.
    """
    fig = go.Figure()

    for name, meta in FACTORY_METADATA.items():
        f_df = df[df["Current_Factory"] == name]
        assigned_count = len(f_df)
        total_sales = float(f_df["Sales"].sum()) if not f_df.empty else 0.0
        color = FACTORY_COLORS.get(name, "#1E293B")

        hover_text = (
            f"<b>🏭 {name}</b><br>"
            f"📍 Location: {meta['city']}, {meta['state']}<br>"
            f"🌐 Coordinates: {meta['latitude']:.4f}° N, {abs(meta['longitude']):.4f}° W<br>"
            f"🏷️ Specialization: {meta['division']}<br>"
            f"📦 Assigned Orders: {assigned_count:,}<br>"
            f"💰 Total Sales: ${total_sales:,.2f}"
        )

        fig.add_trace(
            go.Scattergeo(
                lat=[meta["latitude"]],
                lon=[meta["longitude"]],
                mode="markers+text",
                marker=dict(
                    size=16,
                    symbol="star",
                    color=color,
                    line=dict(width=2, color="#0F172A"),
                ),
                name=f"{name} ({meta['city']}, {meta['state']})",
                text=[f"  {name}"],
                textposition="middle right",
                textfont=dict(size=12, color="#0F172A", family="Arial Black"),
                hoverinfo="text",
                hovertext=hover_text,
            )
        )

    fig.update_geos(
        scope="north america",
        showcountries=True,
        countrycolor="#94A3B8",
        showsubunits=True,
        subunitcolor="#CBD5E1",
        showcoastlines=True,
        coastlinecolor="#94A3B8",
        showland=True,
        landcolor="#F8FAFC",
        showocean=True,
        oceancolor="#EFF6FF",
        showlakes=True,
        lakecolor="#EFF6FF",
        fitbounds="locations",
    )

    fig.update_layout(
        title=dict(
            text="<b>Nassau Candy Manufacturing Facilities Network</b><br><sup>Validated geographic coordinates across 5 production centers in the US</sup>",
            x=0.02,
            font=dict(size=15),
        ),
        margin=dict(l=10, r=10, t=55, b=10),
        height=520,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
    )

    return fig


def aggregate_customer_locations(
    df: pd.DataFrame,
    factory_filter: str = "All Factories",
    product_filter: str = "All Products",
    decision_filter: str = "All Decisions",
) -> pd.DataFrame:
    """
    Filters and aggregates orders by unique customer centroid location
    (Latitude, Longitude, City, State/Province, Country/Region).
    """
    filtered = df.copy()

    if factory_filter != "All Factories":
        filtered = filtered[filtered["Current_Factory"] == factory_filter]

    if product_filter != "All Products":
        filtered = filtered[filtered["Product Name"] == product_filter]

    if decision_filter == "Reallocation Candidates Only (HIGH + REVIEW)":
        filtered = filtered[filtered["Final_Optimization_Decision"].isin(["RELOCATE - HIGH", "RELOCATE - REVIEW"])]
    elif decision_filter == "KEEP CURRENT Only":
        filtered = filtered[filtered["Final_Optimization_Decision"] == "KEEP CURRENT"]
    elif decision_filter != "All Decisions":
        filtered = filtered[filtered["Final_Optimization_Decision"] == decision_filter]

    if filtered.empty:
        return pd.DataFrame()

    agg = (
        filtered.groupby(
            ["Latitude", "Longitude", "City", "State/Province", "Country/Region", "Current_Factory"],
            as_index=False,
        )
        .agg(
            order_count=("Order ID", "count"),
            total_sales=("Sales", "sum"),
            total_units=("Units", "sum"),
            avg_distance_km=("Current_Factory_Distance_KM", "mean"),
            realloc_candidates=(
                "Final_Optimization_Decision",
                lambda s: s.isin(["RELOCATE - HIGH", "RELOCATE - REVIEW"]).sum(),
            ),
        )
    )

    return agg


def build_customer_distribution_map(
    df: pd.DataFrame,
    factory_filter: str = "All Factories",
    product_filter: str = "All Products",
    decision_filter: str = "All Decisions",
) -> go.Figure:
    """
    Creates an interactive nationwide customer distribution map showing demand hubs
    aggregated across postal area centroids, with distinct styling for factories.
    """
    agg = aggregate_customer_locations(
        df=df,
        factory_filter=factory_filter,
        product_filter=product_filter,
        decision_filter=decision_filter,
    )

    fig = go.Figure()

    if not agg.empty:
        max_orders = agg["order_count"].max()
        # Scale marker size smoothly
        sizeref = (2.0 * max_orders) / (22.0 ** 2) if max_orders > 0 else 1.0

        for factory_name in sorted(agg["Current_Factory"].unique()):
            sub_agg = agg[agg["Current_Factory"] == factory_name]
            f_color = FACTORY_COLORS.get(factory_name, "#3B82F6")

            customdata = np.stack(
                (
                    sub_agg["City"],
                    sub_agg["State/Province"],
                    sub_agg["Country/Region"],
                    sub_agg["order_count"],
                    sub_agg["total_sales"],
                    sub_agg["realloc_candidates"],
                    sub_agg["avg_distance_km"],
                ),
                axis=-1,
            )

            hovertemplate = (
                "<b>📍 %{customdata[0]}, %{customdata[1]} (%{customdata[2]})</b><br>"
                "🏭 Current Factory: " + factory_name + "<br>"
                "📦 Order Volume: %{customdata[3]:,} orders<br>"
                "💰 Total Sales: $%{customdata[4]:,.2f}<br>"
                "🔄 Reallocation Candidates: %{customdata[5]:,}<br>"
                "📏 Avg Haversine Distance: %{customdata[6]:,.1f} km<br>"
                "<i>Coords: %{lat:.3f}°, %{lon:.3f}°</i><extra></extra>"
            )

            fig.add_trace(
                go.Scattergeo(
                    lat=sub_agg["Latitude"],
                    lon=sub_agg["Longitude"],
                    mode="markers",
                    marker=dict(
                        size=sub_agg["order_count"],
                        sizemode="area",
                        sizeref=sizeref,
                        sizemin=4,
                        color=f_color,
                        opacity=0.65,
                        line=dict(width=0.75, color="#1E293B"),
                    ),
                    name=f"Demand ({factory_name})",
                    customdata=customdata,
                    hovertemplate=hovertemplate,
                )
            )

    # Add all 5 factories prominently on top
    for name, meta in FACTORY_METADATA.items():
        is_highlighted = (factory_filter == "All Factories") or (factory_filter == name)
        marker_color = FACTORY_COLORS.get(name, "#EF4444")
        marker_size = 18 if is_highlighted else 12
        marker_opacity = 1.0 if is_highlighted else 0.4

        fig.add_trace(
            go.Scattergeo(
                lat=[meta["latitude"]],
                lon=[meta["longitude"]],
                mode="markers+text",
                marker=dict(
                    size=marker_size,
                    symbol="star",
                    color=marker_color,
                    opacity=marker_opacity,
                    line=dict(width=2, color="#0F172A"),
                ),
                name=f"Factory: {name}",
                text=[f"★ {name}" if is_highlighted else ""],
                textposition="bottom center",
                textfont=dict(size=11, color="#0F172A", family="Arial Black"),
                hoverinfo="text",
                hovertext=(
                    f"<b>🏭 FACTORY: {name}</b><br>"
                    f"📍 {meta['city']}, {meta['state']}<br>"
                    f"Specialty: {meta['division']}<br>"
                    f"Coords: {meta['latitude']:.4f}°, {meta['longitude']:.4f}°"
                ),
            )
        )

    fig.update_geos(
        scope="north america",
        showcountries=True,
        countrycolor="#94A3B8",
        showsubunits=True,
        subunitcolor="#CBD5E1",
        showcoastlines=True,
        coastlinecolor="#94A3B8",
        showland=True,
        landcolor="#F8FAFC",
        showocean=True,
        oceancolor="#EFF6FF",
        showlakes=True,
        lakecolor="#EFF6FF",
        fitbounds="locations",
    )

    title_desc = f"Filter: Factory = {factory_filter} | Product = {product_filter} | Decision = {decision_filter}"
    fig.update_layout(
        title=dict(
            text=f"<b>Customer Demand Distribution Across North America</b><br><sup>{title_desc}</sup>",
            x=0.02,
            font=dict(size=15),
        ),
        margin=dict(l=10, r=10, t=55, b=10),
        height=560,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
    )

    return fig


def build_assignment_comparison_map(
    df: pd.DataFrame,
    source_factory: str,
    destination_factory: str,
    candidates_only: bool = False,
    show_connections: bool = True,
    max_lines: int = 40,
) -> go.Figure:
    """
    Renders an assignment comparison map contrasting the source factory
    with the selected destination factory for currently assigned orders.

    Highlights:
    - Source and Destination factory facilities.
    - Customer locations color-coded by whether Haversine distance decreases,
      increases, or remains unchanged.
    - Optional straight-line geographic connections between the proposed
      destination factory and top distance-reduction customer centroids.
    """
    is_valid, err_msg = validate_simulation_inputs(df, source_factory, destination_factory)
    if not is_valid:
        fig = go.Figure()
        fig.add_annotation(text=f"Invalid selection: {err_msg}", showarrow=False, font=dict(size=14, color="red"))
        return fig

    # Run simulation calculation using the validated engine
    sim_result = run_factory_simulation(
        df=df,
        source_factory=source_factory,
        destination_factory=destination_factory,
        candidates_only=candidates_only,
    )

    fig = go.Figure()

    src_meta = FACTORY_METADATA[source_factory]
    dst_meta = FACTORY_METADATA[destination_factory]

    source_col = src_meta["distance_column"]
    dest_col = dst_meta["distance_column"]

    # Filter orders
    orders_df = df[df["Current_Factory"] == source_factory].copy()
    if candidates_only:
        orders_df = orders_df[orders_df["Final_Optimization_Decision"].isin(["RELOCATE - HIGH", "RELOCATE - REVIEW"])]

    if orders_df.empty:
        fig.add_annotation(
            text="No orders match the selected criteria for geographic comparison.",
            showarrow=False,
            font=dict(size=14, color="#64748B"),
        )
        return fig

    # Calculate delta for each order
    cur_d = orders_df[source_col].astype(float)
    dst_d = orders_df[dest_col].astype(float)
    delta_km = cur_d - dst_d  # Positive = reduction / shorter

    # Classify orders using DISTANCE_TOLERANCE_KM
    orders_df["Impact"] = np.where(
        delta_km > DISTANCE_TOLERANCE_KM,
        "Decreased Distance (Shorter)",
        np.where(
            delta_km < -DISTANCE_TOLERANCE_KM,
            "Increased Distance (Longer)",
            "Unchanged Distance",
        ),
    )
    orders_df["Delta_KM"] = delta_km
    orders_df["Cur_KM"] = cur_d
    orders_df["Dst_KM"] = dst_d

    # Aggregate by customer centroid and impact
    agg = (
        orders_df.groupby(
            ["Latitude", "Longitude", "City", "State/Province", "Impact"],
            as_index=False,
        )
        .agg(
            order_count=("Order ID", "count"),
            avg_cur_km=("Cur_KM", "mean"),
            avg_dst_km=("Dst_KM", "mean"),
            avg_saving_km=("Delta_KM", "mean"),
            total_saving_km=("Delta_KM", "sum"),
        )
        .sort_values(by="total_saving_km", ascending=False)
    )

    # 1. Straight-line geographic connection traces (if requested)
    if show_connections:
        # Pick top saving centroids where distance decreased
        saving_hubs = agg[agg["Impact"] == "Decreased Distance (Shorter)"].head(max_lines)

        lats = []
        lons = []
        for _, row in saving_hubs.iterrows():
            # Destination factory to customer centroid
            lats.extend([dst_meta["latitude"], row["Latitude"], None])
            lons.extend([dst_meta["longitude"], row["Longitude"], None])

        if lats:
            fig.add_trace(
                go.Scattergeo(
                    lat=lats,
                    lon=lons,
                    mode="lines",
                    line=dict(width=1.2, color="#059669", dash="dot"),
                    name="Geographic Vector (Geodesic - NOT Truck Route)",
                    hoverinfo="none",
                    opacity=0.55,
                )
            )

    # 2. Add customer centroids grouped by impact
    for impact_label in ["Decreased Distance (Shorter)", "Increased Distance (Longer)", "Unchanged Distance"]:
        sub = agg[agg["Impact"] == impact_label]
        if sub.empty:
            continue

        color = IMPACT_COLORS.get(impact_label, "#94A3B8")
        customdata = np.stack(
            (
                sub["City"],
                sub["State/Province"],
                sub["order_count"],
                sub["avg_cur_km"],
                sub["avg_dst_km"],
                sub["avg_saving_km"],
                sub["total_saving_km"],
            ),
            axis=-1,
        )

        hovertemplate = (
            "<b>📍 %{customdata[0]}, %{customdata[1]}</b><br>"
            "Status: " + impact_label + "<br>"
            "📦 Order Count: %{customdata[2]:,}<br>"
            "Current Distance (Source): %{customdata[3]:,.1f} km<br>"
            "Scenario Distance (Dest): %{customdata[4]:,.1f} km<br>"
            "Avg Distance Shift: %{customdata[5]:+,.1f} km<br>"
            "Total Potential Savings: %{customdata[6]:+,.1f} km<extra></extra>"
        )

        fig.add_trace(
            go.Scattergeo(
                lat=sub["Latitude"],
                lon=sub["Longitude"],
                mode="markers",
                marker=dict(
                    size=sub["order_count"],
                    sizemode="area",
                    sizeref=(2.0 * agg["order_count"].max()) / (20.0 ** 2),
                    sizemin=4,
                    color=color,
                    opacity=0.75,
                    line=dict(width=0.75, color="#1E293B"),
                ),
                name=f"{impact_label} ({len(sub)} hubs)",
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
        )

    # 3. Add Source Factory
    fig.add_trace(
        go.Scattergeo(
            lat=[src_meta["latitude"]],
            lon=[src_meta["longitude"]],
            mode="markers+text",
            marker=dict(size=18, symbol="diamond", color="#EA580C", line=dict(width=2.5, color="#7C2D12")),
            name=f"Source: {source_factory}",
            text=[f"  Source: {source_factory}"],
            textposition="top right",
            textfont=dict(size=12, color="#7C2D12", family="Arial Black"),
            hoverinfo="text",
            hovertext=f"<b>🏭 SOURCE FACTORY: {source_factory}</b><br>{src_meta['city']}, {src_meta['state']}<br>Orders Assigned: {len(orders_df):,}",
        )
    )

    # 4. Add Destination Factory
    fig.add_trace(
        go.Scattergeo(
            lat=[dst_meta["latitude"]],
            lon=[dst_meta["longitude"]],
            mode="markers+text",
            marker=dict(size=20, symbol="star", color="#059669", line=dict(width=2.5, color="#064E3B")),
            name=f"Proposed: {destination_factory}",
            text=[f"  Proposed: {destination_factory}"],
            textposition="bottom right",
            textfont=dict(size=12, color="#064E3B", family="Arial Black"),
            hoverinfo="text",
            hovertext=f"<b>🏭 PROPOSED DESTINATION: {destination_factory}</b><br>{dst_meta['city']}, {dst_meta['state']}<br>Specialty: {dst_meta['division']}",
        )
    )

    fig.update_geos(
        scope="north america",
        showcountries=True,
        countrycolor="#94A3B8",
        showsubunits=True,
        subunitcolor="#CBD5E1",
        showcoastlines=True,
        coastlinecolor="#94A3B8",
        showland=True,
        landcolor="#F8FAFC",
        showocean=True,
        oceancolor="#EFF6FF",
        showlakes=True,
        lakecolor="#EFF6FF",
        fitbounds="locations",
    )

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Geographic Reallocation Analysis: {source_factory} → {destination_factory}</b><br>"
                f"<sup>Green = Shorter Haversine Distance | Red = Longer Haversine Distance | Straight lines = Geodesic Vectors (NOT truck routes)</sup>"
            ),
            x=0.02,
            font=dict(size=14),
        ),
        margin=dict(l=10, r=10, t=55, b=10),
        height=580,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
    )

    return fig


def build_state_concentration_chart(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Builds a horizontal stacked bar chart showing the top states/provinces
    by order volume, broken down by optimization decision.
    """
    top_states = (
        df.groupby("State/Province")["Order ID"]
        .count()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )

    sub_df = df[df["State/Province"].isin(top_states)].copy()

    # Aggregate by State and Decision
    pivot = (
        sub_df.groupby(["State/Province", "Final_Optimization_Decision"])
        .size()
        .unstack(fill_value=0)
    )

    # Sort states by total orders descending (bottom-to-top in horizontal bar)
    pivot["Total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values(by="Total", ascending=True)

    fig = go.Figure()

    for decision in ["KEEP CURRENT", "RELOCATE - REVIEW", "RELOCATE - HIGH"]:
        if decision in pivot.columns:
            color = DECISION_COLORS.get(decision, "#3B82F6")
            fig.add_trace(
                go.Bar(
                    y=pivot.index,
                    x=pivot[decision],
                    name=decision,
                    orientation="h",
                    marker=dict(color=color),
                    hovertemplate="<b>%{y}</b><br>" + decision + ": %{x:,} orders<extra></extra>",
                )
            )

    fig.update_layout(
        barmode="stack",
        title=dict(
            text=f"<b>Top {top_n} States/Provinces by Order Volume & Optimization Status</b>",
            x=0.02,
            font=dict(size=15),
        ),
        xaxis=dict(title="Order Count", gridcolor="#E2E8F0"),
        yaxis=dict(title="State / Province", dtick=1),
        margin=dict(l=10, r=10, t=50, b=40),
        height=480,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    return fig


def build_city_insights_table(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Generates a tabular summary of the top cities with highest potential distance savings,
    highlighting geographic opportunities for managerial reallocation review.
    """
    city_agg = (
        df.groupby(["City", "State/Province", "Country/Region"], as_index=False)
        .agg(
            Total_Orders=("Order ID", "count"),
            Reallocation_Candidates=(
                "Final_Optimization_Decision",
                lambda s: s.isin(["RELOCATE - HIGH", "RELOCATE - REVIEW"]).sum(),
            ),
            Total_Sales=("Sales", "sum"),
            Total_Distance_Saved_KM=("Alternative_Distance_Saving_KM", "sum"),
            Avg_Saving_Pct=("Alternative_Distance_Saving_Pct", "mean"),
            Current_Factory_Mode=("Current_Factory", lambda s: s.mode()[0] if not s.empty else "N/A"),
            Best_Alternative_Mode=("Best_Alternative_Factory", lambda s: s.mode()[0] if not s.empty else "N/A"),
        )
    )

    city_agg["Candidate_Share_Pct"] = (
        (city_agg["Reallocation_Candidates"] / city_agg["Total_Orders"]) * 100.0
    ).round(1)

    top_cities = city_agg.sort_values(by="Total_Distance_Saved_KM", ascending=False).head(top_n)

    display_table = pd.DataFrame({
        "City": top_cities["City"] + ", " + top_cities["State/Province"],
        "Country": top_cities["Country/Region"],
        "Orders": top_cities["Total_Orders"],
        "Realloc Candidates": top_cities["Reallocation_Candidates"],
        "Candidate %": top_cities["Candidate_Share_Pct"].map("{:.1f}%".format),
        "Total Sales ($)": top_cities["Total_Sales"].map("${:,.2f}".format),
        "Potential Distance Saved (KM)": top_cities["Total_Distance_Saved_KM"].map("{:,.0f} km".format),
        "Avg Saving %": top_cities["Avg_Saving_Pct"].map("{:+.1f}%".format),
        "Primary Current Factory": top_cities["Current_Factory_Mode"],
        "Recommended Alternative Factory": top_cities["Best_Alternative_Mode"],
    })

    return display_table


def render_geographic_network_page(df: pd.DataFrame):
    """
    Main Streamlit renderer for the Geographic Network view.
    Integrates Factory Location Map, Customer Centroid Distribution,
    What-If Reallocation Geographic Comparison, and Regional Demand Insights.
    """
    st.markdown('<div class="main-title">🗺️ Geographic Network & Factory Location Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Spatial Supply Chain Analysis, Production Footprint & Reallocation Feasibility | Nassau Candy Distributor</div>',
        unsafe_allow_html=True,
    )

    # Coordinate Validation & Geographic Variation Health Check
    geo_health = validate_geo_coordinates(df)
    if geo_health["has_single_centroid_error"]:
        st.error(
            "❌ Critical Data Anomaly: All customer records share a single national centroid. "
            "Geographic maps cannot accurately reflect distribution."
        )
        st.stop()

    # Informational Centroid Disclaimer Banner
    st.markdown(
        f"""
        <div style="background-color: var(--box-bg, #F8FAFC); border-left: 4px solid #3B82F6; padding: 12px 18px; border-radius: 6px; margin-bottom: 1.25rem; font-size: 0.88rem; color: var(--box-text, #334155); line-height: 1.5;">
            <strong>ℹ️ Geographic Distribution Interpretation:</strong><br>
            • Customer coordinates are validated across <strong>{geo_health['unique_locations']:,} unique regional centroids</strong> 
              (Latitude: {geo_health['lat_min']:.2f}° to {geo_health['lat_max']:.2f}° N, Longitude: {abs(geo_health['lon_min']):.2f}° to {abs(geo_health['lon_max']):.2f}° W) 
              covering the United States and Canada.<br>
            • Customer coordinates represent <strong>postal-code / municipality geographic centroids</strong> rather than exact street-level loading dock addresses.<br>
            • Distances are computed via the <strong>spherical Haversine formula</strong> as direct geometric improvements, not actual commercial truck routes.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # SECTION 1: Factory Location Map & Network Footprint
    # ---------------------------------------------------------
    st.markdown('<div class="section-header">🏭 1. Manufacturing Facilities Network</div>', unsafe_allow_html=True)

    summary_df = get_factory_summary(df)

    # Factory Metric Summary Cards
    f_cols = st.columns(5)
    for idx, (_, row) in enumerate(summary_df.iterrows()):
        with f_cols[idx]:
            st.markdown(
                f"""
                <div class="kpi-card" style="border-top: 3px solid {FACTORY_COLORS.get(row['Factory'], '#3B82F6')};">
                    <div class="kpi-title">{row['Factory']}</div>
                    <div class="kpi-value" style="font-size: 1.35rem;">{row['Assigned Orders']:,}</div>
                    <div class="kpi-subtext">
                        📍 {row['City']}, {row['State']}<br>
                        🏷️ {row['Specialization']}<br>
                        📊 {row['Network Share (%)']:.1f}% network share
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Render Factory Network Map
    factory_map = build_factory_location_map(df)
    st.plotly_chart(factory_map, use_container_width=True)

    with st.expander("📋 View Manufacturing Facilities Specification Table", expanded=False):
        st.dataframe(
            summary_df[[
                "Factory", "City", "State", "Latitude", "Longitude",
                "Specialization", "Assigned Orders", "Network Share (%)", "Total Units", "Total Sales ($)"
            ]],
            use_container_width=True,
            hide_index=True,
        )

    # ---------------------------------------------------------
    # SECTION 2: Customer Distribution Map
    # ---------------------------------------------------------
    st.markdown('<div class="section-header">📍 2. Customer Demand Distribution Across North America</div>', unsafe_allow_html=True)
    st.markdown(
        "Explore nationwide customer demand clusters. Markers represent postal centroids sized proportionally by order volume."
    )

    # Interactive Map Filters
    c_col1, c_col2, c_col3 = st.columns(3)
    with c_col1:
        factory_filter_opt = ["All Factories"] + ALL_FACTORIES
        selected_factory_filter = st.selectbox(
            "Filter Demand by Current Factory",
            options=factory_filter_opt,
            index=0,
            key="geo_factory_filter",
        )
    with c_col2:
        product_list = sorted(df["Product Name"].dropna().unique().tolist())
        product_filter_opt = ["All Products"] + product_list
        selected_product_filter = st.selectbox(
            "Filter Demand by Product (Optional)",
            options=product_filter_opt,
            index=0,
            key="geo_product_filter",
        )
    with c_col3:
        decision_filter_opt = [
            "All Decisions",
            "Reallocation Candidates Only (HIGH + REVIEW)",
            "KEEP CURRENT Only",
            "RELOCATE - HIGH",
            "RELOCATE - REVIEW",
        ]
        selected_decision_filter = st.selectbox(
            "Filter Demand by Recommendation Decision",
            options=decision_filter_opt,
            index=0,
            key="geo_decision_filter",
        )

    customer_map = build_customer_distribution_map(
        df=df,
        factory_filter=selected_factory_filter,
        product_filter=selected_product_filter,
        decision_filter=selected_decision_filter,
    )
    st.plotly_chart(customer_map, use_container_width=True)

    # ---------------------------------------------------------
    # SECTION 3: Current vs. Proposed Assignment Comparison
    # ---------------------------------------------------------
    st.markdown('<div class="section-header">🔄 3. Factory Reassignment Spatial Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        "Evaluate a proposed production reallocation between two manufacturing centers. "
        "Customer centroids are color-coded based on whether the destination factory reduces or increases Haversine distance."
    )

    cmp_col1, cmp_col2, cmp_col3 = st.columns([1.5, 1.5, 1.0])
    with cmp_col1:
        src_fac = st.selectbox(
            "Current (Source) Factory",
            options=ALL_FACTORIES,
            index=0,
            key="geo_cmp_source",
        )
    with cmp_col2:
        available_dest = [f for f in ALL_FACTORIES if f != src_fac]
        dst_fac = st.selectbox(
            "Proposed Destination Factory",
            options=available_dest,
            index=0,
            key="geo_cmp_dest",
        )
    with cmp_col3:
        cmp_scope = st.radio(
            "Order Scope",
            options=["All Orders", "Candidates Only"],
            index=0,
            key="geo_cmp_scope",
        )
        candidates_only_flag = cmp_scope == "Candidates Only"

    # Toggle for straight-line connections
    toggle_col1, toggle_col2 = st.columns([2, 1])
    with toggle_col1:
        show_lines = st.checkbox(
            "Show Straight-Line Geographic Vectors (Top 40 distance-reduction demand hubs)",
            value=True,
            key="geo_show_lines",
            help="Draws direct geodesic lines from the proposed destination factory to top candidate hubs. Not road routes.",
        )
    with toggle_col2:
        inter_dist = get_factory_distance(src_fac, dst_fac)
        st.caption(f"📏 Direct facility separation: **{inter_dist:,.0f} km**")

    # Run validated simulation for KPIs
    sim_stats = run_factory_simulation(
        df=df,
        source_factory=src_fac,
        destination_factory=dst_fac,
        candidates_only=candidates_only_flag,
    )

    # Comparative KPI Cards
    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Orders Evaluated</div>
                <div class="kpi-value">{sim_stats['order_count']:,}</div>
                <div class="kpi-subtext">of {sim_stats['total_source_orders']:,} at {src_fac}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_cols[1]:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Current Distance (KM)</div>
                <div class="kpi-value">{sim_stats['current_total_distance_km']:,.0f}</div>
                <div class="kpi-subtext">Avg: {sim_stats['avg_current_distance_km']:,.0f} km / order</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_cols[2]:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Proposed Distance (KM)</div>
                <div class="kpi-value">{sim_stats['scenario_total_distance_km']:,.0f}</div>
                <div class="kpi-subtext">Avg: {sim_stats['avg_scenario_distance_km']:,.0f} km / order</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_cols[3]:
        saved_km = sim_stats["total_distance_saved_km"]
        sign_char = "+" if saved_km > 0 else ""
        color_style = "color: #10B981;" if saved_km > 0 else ("color: #EF4444;" if saved_km < 0 else "")
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Distance Reduction (KM)</div>
                <div class="kpi-value" style="{color_style}">{sign_char}{saved_km:,.0f}</div>
                <div class="kpi-subtext">{'+' if saved_km > 0 else ''}{sim_stats['pct_reduction_total']:.1f}% net reduction</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi_cols[4]:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Order Distribution</div>
                <div class="kpi-value" style="font-size: 1.25rem;">
                    <span style="color: #10B981;">↓{sim_stats['orders_decreasing']:,}</span> | 
                    <span style="color: #EF4444;">↑{sim_stats['orders_increasing']:,}</span>
                </div>
                <div class="kpi-subtext">Unchanged: {sim_stats['orders_unchanged']:,} orders</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Render Assignment Comparison Map
    cmp_map = build_assignment_comparison_map(
        df=df,
        source_factory=src_fac,
        destination_factory=dst_fac,
        candidates_only=candidates_only_flag,
        show_connections=show_lines,
        max_lines=40,
    )
    st.plotly_chart(cmp_map, use_container_width=True)

    # ---------------------------------------------------------
    # SECTION 4: Regional Concentration & Geographic Insights
    # ---------------------------------------------------------
    st.markdown('<div class="section-header">📊 4. Regional Customer Concentration & Reallocation Opportunities</div>', unsafe_allow_html=True)
    st.markdown(
        "Analyze geographic regions with high order density to identify production reassignment opportunities."
    )

    col_chart, col_table = st.columns([1.2, 1.0])
    with col_chart:
        state_chart = build_state_concentration_chart(df, top_n=15)
        st.plotly_chart(state_chart, use_container_width=True)

    with col_table:
        st.markdown("<strong>Top Cities by Potential Geographic Savings</strong>", unsafe_allow_html=True)
        city_table = build_city_insights_table(df, top_n=10)
        st.dataframe(
            city_table,
            use_container_width=True,
            hide_index=True,
            height=435,
        )

    # ---------------------------------------------------------
    # SECTION 5: Data Quality & Operational Limitations
    # ---------------------------------------------------------
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Geographic Modeling Assumptions & Operational Limitations:</strong><br>
            • <strong>Spherical Haversine Distance:</strong> Distance savings are calculated as great-circle geographic improvements. 
              They do not model interstate highway routes, mountain passes, or road topography (actual ground freight typically exhibits a circuity multiplier of 1.15–1.30×).<br>
            • <strong>Straight-Line Vectors:</strong> Any connecting vectors displayed on maps represent direct geodesic orientation vectors, NOT turn-by-turn truck routes or carrier lanes.<br>
            • <strong>Postal Area Centroids:</strong> Customer coordinates represent geographic centroids of postal codes or cities. True customer delivery addresses may vary within the municipality.<br>
            • <strong>Production & Operational Feasibility:</strong> Moving orders to an alternative factory requires factory confectionery machinery suitability, batch capacity, ingredient sourcing, and shift schedule approval.<br>
            • <strong>No Guaranteed Savings:</strong> Geographic proximity does not guarantee cheaper freight costs, which depend on carrier volume contracts, fuel surcharges, and regional terminal hub pricing.
        </div>
        """,
        unsafe_allow_html=True,
    )
