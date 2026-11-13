"""
Factory Reallocation & Shipping Optimization Recommendation System
Factory Optimization Simulator Module for Nassau Candy Distributor.

Provides simulation modeling, scenario evaluation, geographic distance calculations,
KPI metrics, Plotly visualizations, and Streamlit UI components.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Established factory metadata
FACTORY_METADATA: Dict[str, Dict[str, Any]] = {
    "Lot's O' Nuts": {
        "latitude": 32.881893,
        "longitude": -111.768036,
        "city": "Florence",
        "state": "AZ",
        "division": "Chocolate",
        "distance_column": "Lot's O' Nuts_Distance_KM",
    },
    "Wicked Choccy's": {
        "latitude": 32.076176,
        "longitude": -81.088371,
        "city": "Savannah",
        "state": "GA",
        "division": "Chocolate",
        "distance_column": "Wicked Choccy's_Distance_KM",
    },
    "Sugar Shack": {
        "latitude": 48.119140,
        "longitude": -96.181150,
        "city": "Thief River Falls",
        "state": "MN",
        "division": "Sugar",
        "distance_column": "Sugar Shack_Distance_KM",
    },
    "Secret Factory": {
        "latitude": 41.446333,
        "longitude": -90.565487,
        "city": "Rock Island",
        "state": "IL",
        "division": "Sugar / Other",
        "distance_column": "Secret Factory_Distance_KM",
    },
    "The Other Factory": {
        "latitude": 35.117500,
        "longitude": -89.971107,
        "city": "Memphis",
        "state": "TN",
        "division": "Sugar / Other",
        "distance_column": "The Other Factory_Distance_KM",
    },
}

ALL_FACTORIES = list(FACTORY_METADATA.keys())

# Documented tolerance in kilometers (1 meter) for classifying orders with unchanged distances
DISTANCE_TOLERANCE_KM: float = 0.001


def get_factory_distance(factory1: str, factory2: str) -> float:
    """Calculates the direct geographic Haversine distance in KM between two factories."""
    if factory1 not in FACTORY_METADATA or factory2 not in FACTORY_METADATA:
        raise ValueError(f"Unknown factory: {factory1} or {factory2}")
    if factory1 == factory2:
        return 0.0

    lat1, lon1 = FACTORY_METADATA[factory1]["latitude"], FACTORY_METADATA[factory1]["longitude"]
    lat2, lon2 = FACTORY_METADATA[factory2]["latitude"], FACTORY_METADATA[factory2]["longitude"]

    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return float(R * c)


def validate_simulation_inputs(
    df: pd.DataFrame, source_factory: str, destination_factory: str
) -> Tuple[bool, Optional[str]]:
    """
    Validates source and destination factory selections and ensures required
    distance columns exist in the dataset.
    """
    if source_factory not in FACTORY_METADATA:
        return False, f"Source factory '{source_factory}' is not recognized."
    if destination_factory not in FACTORY_METADATA:
        return False, f"Destination factory '{destination_factory}' is not recognized."
    if source_factory == destination_factory:
        return (
            False,
            f"Source factory and destination factory cannot be identical ('{source_factory}'). Please select different facilities.",
        )

    source_col = FACTORY_METADATA[source_factory]["distance_column"]
    dest_col = FACTORY_METADATA[destination_factory]["distance_column"]

    if source_col not in df.columns:
        return False, f"Source distance column '{source_col}' missing from dataset."
    if dest_col not in df.columns:
        return False, f"Destination distance column '{dest_col}' missing from dataset."

    return True, None


def filter_simulation_orders(
    df: pd.DataFrame,
    source_factory: str,
    destination_factory: str,
    candidates_only: bool = False,
    selected_products: Optional[List[str]] = None,
    min_saving_pct: Optional[float] = None,
) -> pd.DataFrame:
    """
    Filters orders from the selected source factory according to scenario controls:
    - candidates_only: restricts to orders with RELOCATE - HIGH or RELOCATE - REVIEW
    - selected_products: optional subset of product names
    - min_saving_pct: optional minimum potential saving percentage for the selected destination
    """
    mask = df["Current_Factory"] == source_factory

    if candidates_only:
        mask = mask & df["Final_Optimization_Decision"].isin(
            ["RELOCATE - HIGH", "RELOCATE - REVIEW"]
        )

    if selected_products:
        mask = mask & df["Product Name"].isin(selected_products)

    filtered_df = df[mask].copy()

    if min_saving_pct is not None and not filtered_df.empty:
        source_col = FACTORY_METADATA[source_factory]["distance_column"]
        dest_col = FACTORY_METADATA[destination_factory]["distance_column"]
        cur_dist = filtered_df[source_col]
        dest_dist = filtered_df[dest_col]
        saving_pct = ((cur_dist - dest_dist) / np.maximum(cur_dist, 1e-6)) * 100.0
        filtered_df = filtered_df[saving_pct >= min_saving_pct].copy()

    return filtered_df


def run_factory_simulation(
    df: pd.DataFrame,
    source_factory: str,
    destination_factory: str,
    candidates_only: bool = False,
    selected_products: Optional[List[str]] = None,
    min_saving_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Executes a complete simulation evaluating reallocating orders from source_factory
    to destination_factory using the actual destination distance column.

    Returns a structured dictionary of summary KPIs, comparison table, product summary,
    and top-saving orders.
    """
    is_valid, err_msg = validate_simulation_inputs(df, source_factory, destination_factory)
    if not is_valid:
        raise ValueError(err_msg)

    # Filter orders based on user scenario controls
    filtered_df = filter_simulation_orders(
        df=df,
        source_factory=source_factory,
        destination_factory=destination_factory,
        candidates_only=candidates_only,
        selected_products=selected_products,
        min_saving_pct=min_saving_pct,
    )

    source_col = FACTORY_METADATA[source_factory]["distance_column"]
    dest_col = FACTORY_METADATA[destination_factory]["distance_column"]

    total_source_orders = int((df["Current_Factory"] == source_factory).sum())
    order_count = len(filtered_df)

    if order_count == 0:
        empty_comp_cols = [
            "Order ID",
            "City",
            "State/Province",
            "Product Name",
            "Current Factory",
            "Selected Destination Factory",
            "Current Distance (KM)",
            "Scenario Distance (KM)",
            "Distance Change (KM)",
            "Distance Change (%)",
            "Existing Recommendation Decision",
        ]
        return {
            "source_factory": source_factory,
            "destination_factory": destination_factory,
            "candidates_only": candidates_only,
            "selected_products": selected_products,
            "min_saving_pct": min_saving_pct,
            "total_source_orders": total_source_orders,
            "order_count": 0,
            "current_total_distance_km": 0.0,
            "scenario_total_distance_km": 0.0,
            "total_distance_saved_km": 0.0,
            "avg_current_distance_km": 0.0,
            "avg_scenario_distance_km": 0.0,
            "avg_distance_saved_km": 0.0,
            "orders_decreasing": 0,
            "orders_increasing": 0,
            "orders_unchanged": 0,
            "pct_change_total_distance": 0.0,
            "relative_distance_change_pct": 0.0,
            "pct_reduction_total": 0.0,
            "comparison_df": pd.DataFrame(columns=empty_comp_cols),
            "product_summary_df": pd.DataFrame(),
            "top_reductions_df": pd.DataFrame(),
            "is_empty": True,
        }

    # Order-level calculations using the explicit destination column
    current_dist = filtered_df[source_col].astype(float)
    scenario_dist = filtered_df[dest_col].astype(float)
    dist_change_km = current_dist - scenario_dist  # Positive = reduction/saving
    dist_change_pct = (dist_change_km / np.maximum(current_dist, 1e-6)) * 100.0

    current_total_km = float(current_dist.sum())
    scenario_total_km = float(scenario_dist.sum())
    total_saved_km = current_total_km - scenario_total_km

    avg_current_km = float(current_dist.mean())
    avg_scenario_km = float(scenario_dist.mean())
    avg_saved_km = avg_current_km - avg_scenario_km

    orders_decreasing = int((dist_change_km > DISTANCE_TOLERANCE_KM).sum())
    orders_increasing = int((dist_change_km < -DISTANCE_TOLERANCE_KM).sum())
    orders_unchanged = int((np.abs(dist_change_km) <= DISTANCE_TOLERANCE_KM).sum())

    # Total distance reduction percentage: (Current Total - Scenario Total) / Current Total * 100%
    # Positive indicates shorter geographic distance; negative indicates longer distance
    pct_reduction_total = (
        float((total_saved_km / current_total_km) * 100.0) if current_total_km > 0 else 0.0
    )

    # Detailed Comparison Table
    comparison_df = pd.DataFrame(
        {
            "Order ID": filtered_df["Order ID"].values,
            "City": filtered_df["City"].values,
            "State/Province": filtered_df["State/Province"].values,
            "Product Name": filtered_df["Product Name"].values,
            "Current Factory": source_factory,
            "Selected Destination Factory": destination_factory,
            "Current Distance (KM)": current_dist.values,
            "Scenario Distance (KM)": scenario_dist.values,
            "Distance Change (KM)": dist_change_km.values,
            "Distance Change (%)": dist_change_pct.values,
            "Existing Recommendation Decision": filtered_df[
                "Final_Optimization_Decision"
            ].values,
        }
    )

    # Product Summary Table
    product_summary_df = (
        comparison_df.groupby("Product Name")
        .agg(
            Orders=("Order ID", "count"),
            Current_Total_KM=("Current Distance (KM)", "sum"),
            Scenario_Total_KM=("Scenario Distance (KM)", "sum"),
            Distance_Saved_KM=("Distance Change (KM)", "sum"),
            Avg_Distance_Saved_KM=("Distance Change (KM)", "mean"),
        )
        .reset_index()
    )
    product_summary_df["Distance_Saved_Pct"] = (
        product_summary_df["Distance_Saved_KM"]
        / np.maximum(product_summary_df["Current_Total_KM"], 1e-6)
    ) * 100.0
    product_summary_df = product_summary_df.sort_values(
        by="Distance_Saved_KM", ascending=False
    ).reset_index(drop=True)

    # Top 10 orders with largest potential distance reductions
    top_reductions_df = (
        comparison_df.sort_values(by="Distance Change (KM)", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    return {
        "source_factory": source_factory,
        "destination_factory": destination_factory,
        "candidates_only": candidates_only,
        "selected_products": selected_products,
        "min_saving_pct": min_saving_pct,
        "total_source_orders": total_source_orders,
        "order_count": order_count,
        "current_total_distance_km": current_total_km,
        "scenario_total_distance_km": scenario_total_km,
        "total_distance_saved_km": total_saved_km,
        "avg_current_distance_km": avg_current_km,
        "avg_scenario_distance_km": avg_scenario_km,
        "avg_distance_saved_km": avg_saved_km,
        "orders_decreasing": orders_decreasing,
        "orders_increasing": orders_increasing,
        "orders_unchanged": orders_unchanged,
        "pct_change_total_distance": pct_reduction_total,
        "relative_distance_change_pct": pct_reduction_total,
        "pct_reduction_total": pct_reduction_total,
        "comparison_df": comparison_df,
        "product_summary_df": product_summary_df,
        "top_reductions_df": top_reductions_df,
        "is_empty": False,
    }


def render_simulator_page(df: pd.DataFrame):
    """
    Renders the interactive Factory Optimization Simulator page inside Streamlit.
    Provides factory selection, scenario controls, dynamic KPI cards, comparison
    table with CSV download, Plotly charts, and business limitations.
    """
    st.markdown(
        '<div class="main-title">🧪 Factory Optimization Simulator</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Interactive What-If Scenario Analysis for Production Reallocation | Nassau Candy Distributor</div>',
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # 1. Factory Selection & Configuration Controls
    # ---------------------------------------------------------
    st.markdown('<div class="section-header">⚙️ Simulation Parameters & Factory Selection</div>', unsafe_allow_html=True)
    
    col_source, col_dest = st.columns(2)

    with col_source:
        source_factory = st.selectbox(
            "Select Current (Source) Factory",
            options=ALL_FACTORIES,
            index=0,
            key="sim_source_factory",
            help="Choose the production facility currently fulfilling the orders to be evaluated.",
        )

    # Destination options exclude current factory by default
    dest_options = [f for f in ALL_FACTORIES if f != source_factory]
    with col_dest:
        destination_factory = st.selectbox(
            "Select Destination (Scenario) Factory",
            options=dest_options,
            index=0,
            key="sim_dest_factory",
            help="Choose the alternative production facility for what-if simulation (must be different from source).",
        )

    # Display selected factory details & geographic coordinates
    src_meta = FACTORY_METADATA[source_factory]
    dst_meta = FACTORY_METADATA[destination_factory]
    inter_factory_dist = get_factory_distance(source_factory, destination_factory)

    col_meta1, col_meta2, col_meta3 = st.columns([1.5, 1.5, 1.0])
    with col_meta1:
        st.info(
            f"""
            **Source:** **{source_factory}**  
            📍 Location: `{src_meta['city']}, {src_meta['state']}`  
            🌐 Coords: `{src_meta['latitude']:.4f}° N, {src_meta['longitude']:.4f}° W`  
            🏭 Specialty: `{src_meta['division']}`
            """
        )
    with col_meta2:
        st.success(
            f"""
            **Destination:** **{destination_factory}**  
            📍 Location: `{dst_meta['city']}, {dst_meta['state']}`  
            🌐 Coords: `{dst_meta['latitude']:.4f}° N, {dst_meta['longitude']:.4f}° W`  
            🏭 Specialty: `{dst_meta['division']}`
            """
        )
    with col_meta3:
        st.metric(
            "Facility Separation",
            f"{inter_factory_dist:,.0f} km",
            help="Direct geographic Haversine distance between the two manufacturing centers.",
        )

    # ---------------------------------------------------------
    # 2. Scenario Filtering Controls
    # ---------------------------------------------------------
    st.markdown('<div class="section-header">🎯 Scenario Scope & Order Filters</div>', unsafe_allow_html=True)

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        eval_scope = st.radio(
            "Evaluation Order Scope",
            options=[
                "All Orders from Source Factory",
                "Reallocation Candidates Only (RELOCATE - HIGH & REVIEW)",
            ],
            index=0,
            key="sim_eval_scope",
            help="Choose whether to evaluate every order from the source factory or only orders flagged by the optimization model.",
        )
        candidates_only = eval_scope.startswith("Reallocation")

    source_orders = df[df["Current_Factory"] == source_factory]
    available_products = sorted(source_orders["Product Name"].dropna().unique().tolist())

    with filter_col2:
        selected_products = st.multiselect(
            "Filter by Specific Products (Optional)",
            options=available_products,
            default=[],
            key="sim_products",
            help="Leave empty to include all manufactured products from this facility.",
        )

    with filter_col3:
        min_saving_pct = st.slider(
            "Minimum Potential Distance Saving (%)",
            min_value=-100.0,
            max_value=90.0,
            value=-100.0,
            step=5.0,
            key="sim_min_saving_pct",
            help="Filter simulation to orders achieving at least this distance saving percentage (use -100% to evaluate all).",
        )

    # Execute simulation calculation
    sim_results = run_factory_simulation(
        df=df,
        source_factory=source_factory,
        destination_factory=destination_factory,
        candidates_only=candidates_only,
        selected_products=selected_products if selected_products else None,
        min_saving_pct=min_saving_pct if min_saving_pct > -100.0 else None,
    )

    # Scope Summary Banner
    total_source_orders = sim_results["total_source_orders"]
    included_orders = sim_results["order_count"]

    st.markdown(
        f"""
        <div style="background-color: var(--scope-bg, #F1F5F9); border-radius: 8px; padding: 12px 18px; margin: 12px 0 18px 0; border-left: 4px solid var(--scope-border, #0284C7); font-size: 0.95rem; color: var(--scope-text, #0F172A);">
            <strong>Simulation Scope:</strong> Evaluating <strong>{included_orders:,}</strong> orders 
            (out of <strong>{total_source_orders:,}</strong> total orders originated at <em>{source_factory}</em>) 
            simulated for reassignment to <em>{destination_factory}</em>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if sim_results["is_empty"] or included_orders == 0:
        st.warning(
            "⚠️ No orders match the selected scenario controls. "
            "Please broaden your product selection or relax the minimum saving percentage threshold."
        )
        _render_disclaimer_box()
        return

    # ---------------------------------------------------------
    # 3. Dynamic Results KPI Cards
    # ---------------------------------------------------------
    st.markdown('<div class="section-header">📊 Scenario Distance & Operational Impact KPIs</div>', unsafe_allow_html=True)

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Current Total Distance</div>
                <div class="kpi-value">{sim_results['current_total_distance_km']:,.0f} km</div>
                <div class="kpi-subtext">Avg: {sim_results['avg_current_distance_km']:,.1f} km / order</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Scenario Total Distance</div>
                <div class="kpi-value">{sim_results['scenario_total_distance_km']:,.0f} km</div>
                <div class="kpi-subtext">Avg: {sim_results['avg_scenario_distance_km']:,.1f} km / order</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col3:
        saved_km = sim_results["total_distance_saved_km"]
        pct_change = sim_results["pct_reduction_total"]
        saving_color = "#10B981" if saved_km >= 0 else "#EF4444"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Geographic Savings</div>
                <div class="kpi-value" style="color: {saving_color};">{saved_km:+,.0f} km</div>
                <div class="kpi-subtext">Net distance reduction: {pct_change:+.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col4:
        avg_saved = sim_results["avg_distance_saved_km"]
        avg_color = "#10B981" if avg_saved >= 0 else "#EF4444"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Average Distance Saving</div>
                <div class="kpi-value" style="color: {avg_color};">{avg_saved:+,.1f} km</div>
                <div class="kpi-subtext">Per evaluated order ({included_orders:,} orders)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    kpi_col5, kpi_col6, kpi_col7, kpi_col8 = st.columns(4)

    with kpi_col5:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Orders with Decreased Distance</div>
                <div class="kpi-value" style="color: #10B981;">{sim_results['orders_decreasing']:,}</div>
                <div class="kpi-subtext">{((sim_results['orders_decreasing'] / included_orders) * 100):.1f}% of evaluated orders</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col6:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Orders with Increased Distance</div>
                <div class="kpi-value" style="color: #EF4444;">{sim_results['orders_increasing']:,}</div>
                <div class="kpi-subtext">{((sim_results['orders_increasing'] / included_orders) * 100):.1f}% of evaluated orders</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col7:
        unchanged_count = sim_results["orders_unchanged"]
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Unchanged Distance Orders</div>
                <div class="kpi-value">{unchanged_count:,}</div>
                <div class="kpi-subtext">{((unchanged_count / included_orders) * 100):.1f}% neutral (|Δd| ≤ 1 m)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col8:
        pct_red = sim_results["pct_reduction_total"]
        card_color = "#10B981" if pct_red >= 0 else "#EF4444"
        shift_desc = (
            f"Scenario is {pct_red:.1f}% shorter than baseline"
            if pct_red >= 0
            else f"Scenario is {abs(pct_red):.1f}% longer than baseline"
        )
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Distance Reduction Percentage</div>
                <div class="kpi-value" style="color: {card_color};">{pct_red:+.1f}%</div>
                <div class="kpi-subtext">{shift_desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 4. Interactive Plotly Visualizations
    # ---------------------------------------------------------
    st.markdown('<div class="section-header">📈 Scenario Visualizations & Comparative Analytics</div>', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    # Visualization 1: Current vs Scenario Distance Comparison (Aggregated by Product)
    with chart_col1:
        st.subheader("1. Current vs. Scenario Distance by Product")
        prod_df = sim_results["product_summary_df"]
        if not prod_df.empty:
            fig1 = go.Figure()
            fig1.add_trace(
                go.Bar(
                    x=prod_df["Product Name"],
                    y=prod_df["Current_Total_KM"],
                    name=f"Current ({source_factory})",
                    marker_color="#94A3B8",
                    hovertemplate="%{x}<br>Current: %{y:,.0f} km<extra></extra>",
                )
            )
            fig1.add_trace(
                go.Bar(
                    x=prod_df["Product Name"],
                    y=prod_df["Scenario_Total_KM"],
                    name=f"Scenario ({destination_factory})",
                    marker_color="#2563EB",
                    hovertemplate="%{x}<br>Scenario: %{y:,.0f} km<extra></extra>",
                )
            )
            fig1.update_layout(
                barmode="group",
                title=f"Total Distance (KM): {source_factory} vs {destination_factory}",
                xaxis_title="Product Name",
                yaxis_title="Total Haversine Distance (KM)",
                template="plotly_white",
                height=420,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(tickangle=-40),
            )
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("No product comparison data available for this selection.")

    # Visualization 2: Distribution of Per-Order Distance Changes
    with chart_col2:
        st.subheader("2. Distribution of Distance Changes (KM)")
        comp_df = sim_results["comparison_df"]
        fig2 = px.histogram(
            comp_df,
            x="Distance Change (KM)",
            nbins=35,
            color_discrete_sequence=["#0284C7"],
            title="Distribution of Order Distance Savings (Positive = Reduction)",
            labels={"Distance Change (KM)": "Distance Reduction (KM)"},
            template="plotly_white",
        )
        fig2.add_vline(
            x=0,
            line_dash="dash",
            line_color="#EF4444",
            annotation_text="Zero Shift Baseline",
            annotation_position="top left",
        )
        fig2.update_layout(
            height=420,
            yaxis_title="Order Frequency",
            xaxis_title="Potential Geographic Distance Saved (KM)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)

    # Visualization 3: Top 10 Orders with Largest Potential Distance Reductions
    with chart_col3:
        st.subheader("3. Top 10 Orders with Largest Distance Reductions")
        top_df = sim_results["top_reductions_df"]
        if not top_df.empty:
            top_display = top_df.copy()
            top_display["Label"] = (
                top_display["Order ID"].astype(str)
                + " ("
                + top_display["City"]
                + ", "
                + top_display["State/Province"]
                + ")"
            )
            top_display = top_display.sort_values(by="Distance Change (KM)", ascending=True)

            fig3 = px.bar(
                top_display,
                x="Distance Change (KM)",
                y="Label",
                orientation="h",
                color="Distance Change (KM)",
                color_continuous_scale="Tealgrn",
                title="Top 10 Orders by Geographic Distance Reduction (KM)",
                labels={"Distance Change (KM)": "Distance Saved (KM)", "Label": "Order & Destination"},
                template="plotly_white",
            )
            fig3.update_layout(height=420, coloraxis_showscale=False)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No top reductions available.")

    # Visualization 4: Product-Wise Savings Summary
    with chart_col4:
        st.subheader("4. Product-Wise Geographic Savings Summary")
        if not prod_df.empty:
            fig4 = px.bar(
                prod_df,
                x="Distance_Saved_KM",
                y="Product Name",
                orientation="h",
                color="Distance_Saved_KM",
                color_continuous_scale=["#EF4444", "#F59E0B", "#10B981"],
                title="Total Potential Distance Reduction by Product (KM)",
                labels={"Distance_Saved_KM": "Total Distance Saved (KM)"},
                template="plotly_white",
            )
            fig4.update_layout(height=420, coloraxis_showscale=False)
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No product summary chart available.")

    # ---------------------------------------------------------
    # 5. Product Summary & Detailed Comparison Table
    # ---------------------------------------------------------
    st.markdown('<div class="section-header">📋 Detailed Scenario Comparison Table</div>', unsafe_allow_html=True)
    st.markdown(
        f"Individual order comparison evaluating transfer from **{source_factory}** to **{destination_factory}**. "
        "Showing order-level baseline versus simulated scenario distances."
    )

    comp_df_display = sim_results["comparison_df"].copy()
    
    # Format numeric columns for presentation
    formatted_df = comp_df_display.copy()
    formatted_df["Current Distance (KM)"] = formatted_df["Current Distance (KM)"].map("{:,.1f}".format)
    formatted_df["Scenario Distance (KM)"] = formatted_df["Scenario Distance (KM)"].map("{:,.1f}".format)
    formatted_df["Distance Change (KM)"] = formatted_df["Distance Change (KM)"].map("{:+,.1f}".format)
    formatted_df["Distance Change (%)"] = formatted_df["Distance Change (%)"].map("{:+.1f}%".format)

    st.dataframe(
        formatted_df,
        use_container_width=True,
        hide_index=True,
    )

    # CSV Download Button
    csv_data = comp_df_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download Scenario Comparison Table as CSV ({len(comp_df_display):,} Records)",
        data=csv_data,
        file_name=f"simulation_{source_factory.lower().replace(' ', '_')}_to_{destination_factory.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        key="btn_download_sim_csv",
    )

    # ---------------------------------------------------------
    # 6. Business Limitations & Decision Support Disclaimers
    # ---------------------------------------------------------
    _render_disclaimer_box()


def _render_disclaimer_box():
    """Renders business limitations and methodology boundaries."""
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Factory Simulation Methodology & Business Limitations:</strong><br>
            • <strong>Haversine Approximation:</strong> All distance figures represent straight-line geographic distances 
              calculated using the spherical Haversine formula ($R = 6,371.0\\text{ km}$). They serve as proxy measures 
              and do <em>not</em> reflect actual highway, rail, intermodal route miles, or road geography.<br>
            • <strong>Potential vs. Guaranteed Logistics Savings:</strong> A positive geographic distance reduction in this simulator 
              indicates spatial proximity improvement. It does <em>not</em> prove guaranteed freight cost reductions, 
              faster transit lead times, or lower carrier pricing.<br>
            • <strong>Manufacturing Feasibility & Factory Capacity:</strong> Production reallocation must account for factory machine tooling, 
              facility capacity ceilings, labor availability, ingredient supply chains, and batch lead times before physical reassignment. 
              No production compatibility is assumed without operational shop-floor validation.<br>
            • <strong>Decision Support Framework:</strong> This what-if simulation tool provides heuristic decision support for logistics 
              planners and operations executives. Recommendation scores and risk tiers are decision-support heuristics 
              (Established Baseline: <code>KEEP CURRENT = 5,873</code>, <code>RELOCATE - REVIEW = 4,105</code>, <code>RELOCATE - HIGH = 216</code>). 
              It does not replace contractual carrier rate schedules or supply chain ERP execution.
        </div>
        """,
        unsafe_allow_html=True,
    )
