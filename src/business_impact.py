"""
Factory Reallocation & Shipping Optimization Recommendation System
Business Impact & Decision Intelligence Module for Nassau Candy Distributor.

Provides business impact modeling, transparent freight-cost scenario simulation,
decision prioritization with multi-factor driver explanations, operational risk
and manufacturing compatibility checking, and human-reviewer action roadmaps.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Reuse established metadata, tolerance, and simulation engine from src.simulator
from src.simulator import (
    FACTORY_METADATA,
    ALL_FACTORIES,
    DISTANCE_TOLERANCE_KM,
    get_factory_distance,
    run_factory_simulation,
    validate_simulation_inputs,
)

# Standard division compatibility rules
# Maps factory specializations to supported product divisions
FACTORY_SPECIALIZATION_MAP: Dict[str, List[str]] = {
    "Lot's O' Nuts": ["Chocolate"],
    "Wicked Choccy's": ["Chocolate"],
    "Sugar Shack": ["Sugar"],
    "Secret Factory": ["Sugar", "Other"],
    "The Other Factory": ["Sugar", "Other"],
}


def calculate_freight_cost_proxy(
    current_distance_km: float,
    scenario_distance_km: float,
    cost_per_km: float,
) -> Dict[str, float]:
    """
    Computes illustrative freight cost proxies from straight-line distances
    using a transparent, user-supplied assumed cost per kilometer.

    IMPORTANT: This is a geometric distance-based cost proxy for scenario
    modeling, NOT an actual carrier invoice, contract rate, or guaranteed saving.
    """
    if cost_per_km < 0:
        raise ValueError(f"Assumed cost per kilometer must be non-negative, got: {cost_per_km}")

    baseline_proxy = float(current_distance_km * cost_per_km)
    scenario_proxy = float(scenario_distance_km * cost_per_km)
    cost_diff = float(baseline_proxy - scenario_proxy)  # Positive = reduction / saving

    pct_diff = (
        float((cost_diff / baseline_proxy) * 100.0)
        if baseline_proxy > 0
        else 0.0
    )

    return {
        "assumed_cost_per_km": float(cost_per_km),
        "baseline_cost_proxy": baseline_proxy,
        "scenario_cost_proxy": scenario_proxy,
        "estimated_difference": cost_diff,
        "pct_difference": pct_diff,
    }


def check_operational_compatibility(
    product_division: str,
    target_factory: str,
) -> Dict[str, Any]:
    """
    Evaluates whether the target production facility specializes in the product's division.

    Returns:
        dict: Compatibility classification ('COMPATIBLE', 'REVIEW_REQUIRED', 'INCOMPATIBLE'),
              warning message, and recommended verification check.
    """
    if target_factory not in FACTORY_SPECIALIZATION_MAP:
        return {
            "status": "UNKNOWN",
            "is_compatible": False,
            "badge_color": "#64748B",
            "message": f"Factory '{target_factory}' specialization unknown.",
            "action": "Verify plant machine configuration with operations team.",
        }

    supported_divisions = FACTORY_SPECIALIZATION_MAP[target_factory]
    prod_div = str(product_division).strip()

    if prod_div in supported_divisions:
        return {
            "status": "COMPATIBLE",
            "is_compatible": True,
            "badge_color": "#10B981",
            "message": f"Division Aligned: {target_factory} specializes in {', '.join(supported_divisions)}.",
            "action": "Standard batch capacity and schedule verification required.",
        }

    # Partial / Multi-division facilities
    if "Other" in supported_divisions and prod_div != "Chocolate":
        return {
            "status": "REVIEW_REQUIRED",
            "is_compatible": True,
            "badge_color": "#F59E0B",
            "message": f"Secondary Alignment: {target_factory} handles Sugar/Other; line tooling requires check.",
            "action": "Inspect confectionery machinery tooling and switchover time.",
        }

    # Cross-division mismatch (e.g. Chocolate product proposed for Sugar-only plant)
    return {
        "status": "INCOMPATIBLE",
        "is_compatible": False,
        "badge_color": "#EF4444",
        "message": f"Cross-Division Mismatch: Product is '{prod_div}', but {target_factory} specializes in {', '.join(supported_divisions)}.",
        "action": "Major retooling, confectionery line adaptation, and allergen controls required.",
    }


def generate_recommendation_drivers(row: pd.Series) -> str:
    """
    Generates a human-readable explanation of why an order scored high on the
    heuristic Recommendation Score (composed of 40% Distance, 25% Burden,
    20% Margin, 15% Volume).
    """
    drivers = []

    dist_score = row.get("Distance_Score", 0.0)
    saving_km = row.get("Alternative_Distance_Saving_KM", 0.0)
    saving_pct = row.get("Alternative_Distance_Saving_Pct", 0.0)
    burden_score = row.get("Burden_Reduction_Score", 0.0)
    profit_margin = row.get("Profit_Margin_Pct", 0.0)
    units = row.get("Units", 0)

    if saving_pct >= 60.0 or dist_score >= 70.0:
        drivers.append(f"Significant distance reduction ({saving_pct:.0f}% / {saving_km:,.0f} km)")
    elif saving_km > 500.0:
        drivers.append(f"Proximity gain ({saving_km:,.0f} km closer)")

    if burden_score >= 20.0:
        drivers.append("High facility burden relief")

    if profit_margin >= 50.0:
        drivers.append(f"High product margin ({profit_margin:.1f}%)")

    if units >= 10:
        drivers.append(f"High order volume ({units} units)")

    if not drivers:
        drivers.append("Balanced multi-criteria score")

    return "; ".join(drivers)


def run_global_business_impact(
    df: pd.DataFrame,
    candidates_only: bool = True,
    selected_products: Optional[List[str]] = None,
    assumed_cost_per_km: float = 1.50,
) -> Dict[str, Any]:
    """
    Evaluates network-wide business impact of reassigning evaluated orders to
    their recommended Best_Alternative_Factory.

    Preserves sign conventions:
    - Distance saved = Current distance - Best alternative distance
    - Positive = Geographic reduction / proximity improvement
    - Negative = Additional geographic distance
    """
    filtered_df = df.copy()

    if candidates_only:
        filtered_df = filtered_df[
            filtered_df["Final_Optimization_Decision"].isin(
                ["RELOCATE - HIGH", "RELOCATE - REVIEW"]
            )
        ]

    if selected_products:
        filtered_df = filtered_df[filtered_df["Product Name"].isin(selected_products)]

    order_count = len(filtered_df)
    total_dataset_orders = len(df)

    if order_count == 0:
        return {
            "order_count": 0,
            "total_dataset_orders": total_dataset_orders,
            "realloc_candidates": 0,
            "high_priority_candidates": 0,
            "current_total_distance_km": 0.0,
            "scenario_total_distance_km": 0.0,
            "total_distance_saved_km": 0.0,
            "avg_current_distance_km": 0.0,
            "avg_scenario_distance_km": 0.0,
            "avg_distance_saved_km": 0.0,
            "orders_decreasing": 0,
            "orders_increasing": 0,
            "orders_unchanged": 0,
            "aggregate_reduction_pct": 0.0,
            "mean_order_saving_pct": 0.0,
            "freight_proxy": calculate_freight_cost_proxy(0.0, 0.0, assumed_cost_per_km),
            "prioritized_df": pd.DataFrame(),
            "compatibility_summary": {},
            "is_empty": True,
        }

    # Distances
    cur_dist = filtered_df["Current_Factory_Distance_KM"].astype(float)
    scen_dist = filtered_df["Best_Alternative_Distance_KM"].astype(float)
    delta_km = cur_dist - scen_dist  # Positive = reduction / closer

    cur_total_km = float(cur_dist.sum())
    scen_total_km = float(scen_dist.sum())
    saved_total_km = cur_total_km - scen_total_km

    avg_cur_km = float(cur_dist.mean())
    avg_scen_km = float(scen_dist.mean())
    avg_saved_km = avg_cur_km - avg_scen_km

    orders_decreasing = int((delta_km > DISTANCE_TOLERANCE_KM).sum())
    orders_increasing = int((delta_km < -DISTANCE_TOLERANCE_KM).sum())
    orders_unchanged = int((delta_km.abs() <= DISTANCE_TOLERANCE_KM).sum())

    # Aggregate percentage reduction vs order-level mean
    agg_red_pct = (
        float((saved_total_km / cur_total_km) * 100.0) if cur_total_km > 0 else 0.0
    )
    mean_order_pct = float(filtered_df["Alternative_Distance_Saving_Pct"].mean())

    # Candidate counts
    realloc_candidates = int(
        filtered_df["Final_Optimization_Decision"]
        .isin(["RELOCATE - HIGH", "RELOCATE - REVIEW"])
        .sum()
    )
    high_priority = int(
        (filtered_df["Final_Optimization_Decision"] == "RELOCATE - HIGH").sum()
    )

    # Illustrative freight cost proxy
    freight_proxy = calculate_freight_cost_proxy(
        cur_total_km, scen_total_km, assumed_cost_per_km
    )

    # Operational compatibility check for each order
    compat_results = []
    driver_explanations = []

    for _, row in filtered_df.iterrows():
        p_div = row.get("Division", "Other")
        target_f = row.get("Best_Alternative_Factory", "")
        chk = check_operational_compatibility(p_div, target_f)
        compat_results.append(chk["status"])
        driver_explanations.append(generate_recommendation_drivers(row))

    filtered_df = filtered_df.copy()
    filtered_df["Operational_Compatibility"] = compat_results
    filtered_df["Key_Decision_Drivers"] = driver_explanations
    filtered_df["Illustrative_Cost_Difference_$"] = delta_km * assumed_cost_per_km

    # Decision-ready prioritized table
    priority_cols = [
        "Order ID",
        "City",
        "State/Province",
        "Product Name",
        "Division",
        "Current_Factory",
        "Best_Alternative_Factory",
        "Current_Factory_Distance_KM",
        "Best_Alternative_Distance_KM",
        "Alternative_Distance_Saving_KM",
        "Alternative_Distance_Saving_Pct",
        "Sales",
        "Units",
        "Gross Profit",
        "Profit_Margin_Pct",
        "Recommendation_Score",
        "Final_Optimization_Decision",
        "Optimization_Risk",
        "Operational_Compatibility",
        "Illustrative_Cost_Difference_$",
        "Key_Decision_Drivers",
    ]

    prioritized_df = filtered_df[priority_cols].sort_values(
        by=["Recommendation_Score", "Alternative_Distance_Saving_KM"],
        ascending=[False, False],
    ).reset_index(drop=True)

    compat_counts = pd.Series(compat_results).value_counts().to_dict()

    return {
        "order_count": order_count,
        "total_dataset_orders": total_dataset_orders,
        "realloc_candidates": realloc_candidates,
        "high_priority_candidates": high_priority,
        "current_total_distance_km": cur_total_km,
        "scenario_total_distance_km": scen_total_km,
        "total_distance_saved_km": saved_total_km,
        "avg_current_distance_km": avg_cur_km,
        "avg_scenario_distance_km": avg_scen_km,
        "avg_distance_saved_km": avg_saved_km,
        "orders_decreasing": orders_decreasing,
        "orders_increasing": orders_increasing,
        "orders_unchanged": orders_unchanged,
        "aggregate_reduction_pct": agg_red_pct,
        "mean_order_saving_pct": mean_order_pct,
        "freight_proxy": freight_proxy,
        "prioritized_df": prioritized_df,
        "compatibility_summary": compat_counts,
        "is_empty": False,
    }


def render_business_impact_page(df: pd.DataFrame):
    """
    Renders the Business Impact & Decision Intelligence view in Streamlit.
    """
    st.markdown(
        '<div class="main-title">📈 Business Impact & Decision Intelligence</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Multi-Criteria Relocation Evaluation, Illustrative Freight-Cost Modeling & Operational Readiness | Nassau Candy Distributor</div>',
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # 1. Scope Selection & Freight Cost Parameter Controls
    # ---------------------------------------------------------
    st.markdown('<div class="section-header">⚙️ Scenario Configuration & Assumptions</div>', unsafe_allow_html=True)

    cfg_col1, cfg_col2, cfg_col3 = st.columns([1.5, 1.5, 1.2])

    with cfg_col1:
        scenario_mode = st.radio(
            "Evaluation Scenario Mode",
            options=[
                "Global Network Optimization (All Recommended Transfers)",
                "Facility-to-Facility Reallocation (Pairwise What-If)",
            ],
            index=0,
            key="bi_scenario_mode",
            help="Choose between evaluating all network-wide recommended reassignments or simulating a specific pairwise factory transfer.",
        )

    with cfg_col2:
        candidate_scope = st.radio(
            "Target Candidate Population",
            options=[
                "Reallocation Candidates Only (HIGH + REVIEW)",
                "All Orders in Scope",
            ],
            index=0,
            key="bi_candidate_scope",
            help="Focus on flagged relocation candidates (Recommendation Score ≥ 50) or evaluate the entire active order volume.",
        )
        candidates_only = candidate_scope.startswith("Reallocation")

    with cfg_col3:
        cost_per_km = st.number_input(
            "Assumed Cost per Kilometer ($/km)",
            min_value=0.0,
            max_value=15.0,
            value=1.50,
            step=0.25,
            format="%.2f",
            key="bi_cost_per_km",
            help="Enter an assumed freight shipping cost per kilometer for illustrative proxy estimation. NOT an actual carrier contract rate.",
        )

    # Pairwise controls if Mode 2 selected
    is_pairwise = scenario_mode.startswith("Facility-to-Facility")
    source_factory = ALL_FACTORIES[0]
    dest_factory = ALL_FACTORIES[1]

    if is_pairwise:
        pair_col1, pair_col2 = st.columns(2)
        with pair_col1:
            source_factory = st.selectbox(
                "Select Source Factory",
                options=ALL_FACTORIES,
                index=0,
                key="bi_source_factory",
            )
        dest_options = [f for f in ALL_FACTORIES if f != source_factory]
        with pair_col2:
            dest_factory = st.selectbox(
                "Select Destination Scenario Factory",
                options=dest_options,
                index=0,
                key="bi_dest_factory",
            )

    # Product Filter
    available_products = sorted(df["Product Name"].dropna().unique().tolist())
    selected_products = st.multiselect(
        "Filter by Specific Products (Optional)",
        options=available_products,
        default=[],
        key="bi_products",
        help="Leave blank to include all confectionery products.",
    )

    # Compute impact based on selected mode
    if not is_pairwise:
        # Global Network Mode
        impact_results = run_global_business_impact(
            df=df,
            candidates_only=candidates_only,
            selected_products=selected_products if selected_products else None,
            assumed_cost_per_km=cost_per_km,
        )
    else:
        # Pairwise Mode: Reuse run_factory_simulation
        sim_res = run_factory_simulation(
            df=df,
            source_factory=source_factory,
            destination_factory=dest_factory,
            candidates_only=candidates_only,
            selected_products=selected_products if selected_products else None,
        )

        freight_proxy = calculate_freight_cost_proxy(
            sim_res["current_total_distance_km"],
            sim_res["scenario_total_distance_km"],
            cost_per_km,
        )

        # Build prioritized dataframe for pairwise
        comp_df = sim_res["comparison_df"].copy()
        if not comp_df.empty:
            # Join with df to get financial fields
            merge_cols = ["Order ID", "Division", "Sales", "Units", "Gross Profit", "Profit_Margin_Pct", "Recommendation_Score", "Optimization_Risk"]
            merged = comp_df.merge(df[merge_cols].drop_duplicates(subset=["Order ID"]), on="Order ID", how="left")

            compat_list = []
            driver_list = []
            for _, r in merged.iterrows():
                p_div = r.get("Division", "Other")
                compat_list.append(check_operational_compatibility(p_div, dest_factory)["status"])
                driver_list.append(generate_recommendation_drivers(r))

            merged["Operational_Compatibility"] = compat_list
            merged["Key_Decision_Drivers"] = driver_list
            merged["Illustrative_Cost_Difference_$"] = merged["Distance Change (KM)"] * cost_per_km
            prioritized_df = merged.sort_values(by="Distance Change (KM)", ascending=False).reset_index(drop=True)
            compat_counts = pd.Series(compat_list).value_counts().to_dict()
        else:
            prioritized_df = pd.DataFrame()
            compat_counts = {}

        if not comp_df.empty:
            pairwise_candidates = int(
                merged["Final_Optimization_Decision"]
                .isin(["RELOCATE - HIGH", "RELOCATE - REVIEW"])
                .sum()
            )
            pairwise_high_priority = int(
                (merged["Final_Optimization_Decision"] == "RELOCATE - HIGH").sum()
            )
        else:
            pairwise_candidates = 0
            pairwise_high_priority = 0

        impact_results = {
            "order_count": sim_res["order_count"],
            "total_dataset_orders": len(df),
            "realloc_candidates": pairwise_candidates,
            "high_priority_candidates": pairwise_high_priority,
            "current_total_distance_km": sim_res["current_total_distance_km"],
            "scenario_total_distance_km": sim_res["scenario_total_distance_km"],
            "total_distance_saved_km": sim_res["total_distance_saved_km"],
            "avg_current_distance_km": sim_res["avg_current_distance_km"],
            "avg_scenario_distance_km": sim_res["avg_scenario_distance_km"],
            "avg_distance_saved_km": sim_res["avg_distance_saved_km"],
            "orders_decreasing": sim_res["orders_decreasing"],
            "orders_increasing": sim_res["orders_increasing"],
            "orders_unchanged": sim_res["orders_unchanged"],
            "aggregate_reduction_pct": sim_res["pct_reduction_total"],
            "mean_order_saving_pct": float(comp_df["Distance Change (%)"].mean()) if not comp_df.empty else 0.0,
            "freight_proxy": freight_proxy,
            "prioritized_df": prioritized_df,
            "compatibility_summary": compat_counts,
            "is_empty": sim_res["is_empty"],
        }

    # Scope Summary Banner
    st.markdown(
        f"""
        <div style="background-color: var(--scope-bg, #F1F5F9); border-radius: 8px; padding: 12px 18px; margin: 12px 0 18px 0; border-left: 4px solid var(--scope-border, #0284C7); font-size: 0.95rem; color: var(--scope-text, #0F172A);">
            <strong>Active Scenario:</strong> Evaluating <strong>{impact_results['order_count']:,} orders</strong> 
            ({'Candidates only' if candidates_only else 'All orders in scope'}) | 
            Assumed Shipping Cost: <strong>${cost_per_km:.2f} / km</strong> (Illustrative Proxy)
        </div>
        """,
        unsafe_allow_html=True,
    )

    if impact_results["is_empty"] or impact_results["order_count"] == 0:
        st.warning("⚠️ No orders match the selected criteria. Please broaden your scope or product filter.")
        _render_business_disclaimer()
        return

    # ---------------------------------------------------------
    # 2. Business Impact Overview & Freight-Cost Scenario KPIs
    # ---------------------------------------------------------
    st.markdown('<div class="section-header">📊 1. Business Impact Overview & Geographic Distance Shifts</div>', unsafe_allow_html=True)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Evaluated Orders</div>
                <div class="kpi-value">{impact_results['order_count']:,}</div>
                <div class="kpi-subtext">{(impact_results['order_count'] / len(df) * 100):.1f}% of total network orders</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Current Total Distance</div>
                <div class="kpi-value">{impact_results['current_total_distance_km']:,.0f} km</div>
                <div class="kpi-subtext">Avg: {impact_results['avg_current_distance_km']:,.1f} km / order</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Proposed Scenario Distance</div>
                <div class="kpi-value">{impact_results['scenario_total_distance_km']:,.0f} km</div>
                <div class="kpi-subtext">Avg: {impact_results['avg_scenario_distance_km']:,.1f} km / order</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi4:
        tot_saved = impact_results["total_distance_saved_km"]
        color_tot = "#10B981" if tot_saved >= 0 else "#EF4444"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Geographic Savings</div>
                <div class="kpi-value" style="color: {color_tot};">{tot_saved:+,.0f} km</div>
                <div class="kpi-subtext">Net aggregate reduction: {impact_results['aggregate_reduction_pct']:+.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Row 2: Partition Counts and Order-Level vs Aggregate Distinction
    kpi5, kpi6, kpi7, kpi8 = st.columns(4)

    with kpi5:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Orders with Reduced Distance</div>
                <div class="kpi-value" style="color: #10B981;">{impact_results['orders_decreasing']:,}</div>
                <div class="kpi-subtext">{((impact_results['orders_decreasing'] / impact_results['order_count']) * 100):.1f}% closer (Δd > 1 m)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi6:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Orders with Increased Distance</div>
                <div class="kpi-value" style="color: #EF4444;">{impact_results['orders_increasing']:,}</div>
                <div class="kpi-subtext">{((impact_results['orders_increasing'] / impact_results['order_count']) * 100):.1f}% longer (Δd < -1 m)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi7:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Unchanged Distance Orders</div>
                <div class="kpi-value">{impact_results['orders_unchanged']:,}</div>
                <div class="kpi-subtext">{((impact_results['orders_unchanged'] / impact_results['order_count']) * 100):.1f}% neutral (|Δd| ≤ 1 m)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi8:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Distance Reduction Percentage</div>
                <div class="kpi-value">{impact_results['aggregate_reduction_pct']:+.1f}%</div>
                <div class="kpi-subtext">Aggregate net (Order-level mean: {impact_results['mean_order_saving_pct']:+.1f}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------------------------------------------------
    # 3. Transparent Freight-Cost Scenario Model
    # ---------------------------------------------------------
    st.markdown('<div class="section-header">💵 2. Transparent Freight-Cost Scenario Proxy</div>', unsafe_allow_html=True)
    st.markdown(
        "Illustrative cost proxies computed as `Distance (km) × Assumed Rate ($/km)`. "
        "**Disclaimer:** These figures are geometric distance-based cost proxies for managerial review, "
        "**NOT guaranteed carrier savings or audited logistics invoices**."
    )

    f_proxy = impact_results["freight_proxy"]
    c_col1, c_col2, c_col3, c_col4 = st.columns(4)

    with c_col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Assumed Rate Benchmark</div>
                <div class="kpi-value">${f_proxy['assumed_cost_per_km']:.2f} / km</div>
                <div class="kpi-subtext">User-selected parameter</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Baseline Cost Proxy</div>
                <div class="kpi-value">${f_proxy['baseline_cost_proxy']:,.2f}</div>
                <div class="kpi-subtext">Current distance @ assumed rate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Scenario Cost Proxy</div>
                <div class="kpi-value">${f_proxy['scenario_cost_proxy']:,.2f}</div>
                <div class="kpi-subtext">Reassigned distance @ assumed rate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_col4:
        diff_val = f_proxy["estimated_difference"]
        color_diff = "#10B981" if diff_val >= 0 else "#EF4444"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Illustrative Difference</div>
                <div class="kpi-value" style="color: {color_diff};">{'+' if diff_val >= 0 else ''}${diff_val:,.2f}</div>
                <div class="kpi-subtext">{f_proxy['pct_difference']:+.1f}% proxy difference</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------------------------------------------------
    # 4. Interactive Visualizations
    # ---------------------------------------------------------
    st.markdown('<div class="section-header">📈 3. Decision Intelligence Analytics</div>', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Operational Compatibility Distribution")
        compat_dict = impact_results["compatibility_summary"]
        if compat_dict:
            compat_df = pd.DataFrame([
                {"Compatibility": k, "Orders": v} for k, v in compat_dict.items()
            ])
            color_map = {
                "COMPATIBLE": "#10B981",
                "REVIEW_REQUIRED": "#F59E0B",
                "INCOMPATIBLE": "#EF4444",
                "UNKNOWN": "#64748B",
            }
            fig_compat = px.pie(
                compat_df,
                names="Compatibility",
                values="Orders",
                color="Compatibility",
                color_discrete_map=color_map,
                hole=0.45,
                title="Orders by Facility-Product Division Compatibility",
            )
            fig_compat.update_traces(textposition="inside", textinfo="percent+label")
            fig_compat.update_layout(height=370, margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_compat, use_container_width=True)
        else:
            st.info("No compatibility data available.")

    with chart_col2:
        st.subheader("Top Confectionery Products by Illustrative Difference")
        p_df = impact_results["prioritized_df"]
        if not p_df.empty and "Product Name" in p_df.columns:
            top_prod = (
                p_df.groupby("Product Name")["Illustrative_Cost_Difference_$"]
                .sum()
                .reset_index()
                .sort_values(by="Illustrative_Cost_Difference_$", ascending=True)
                .tail(10)
            )
            fig_prod = px.bar(
                top_prod,
                x="Illustrative_Cost_Difference_$",
                y="Product Name",
                orientation="h",
                color="Illustrative_Cost_Difference_$",
                color_continuous_scale="Tealgrn",
                title=f"Top 10 Products by Cost Proxy Difference (@ ${cost_per_km:.2f}/km)",
                labels={"Illustrative_Cost_Difference_$": "Cost Proxy Difference ($)"},
            )
            fig_prod.update_layout(height=370, margin=dict(t=40, b=20, l=20, r=20), coloraxis_showscale=False)
            st.plotly_chart(fig_prod, use_container_width=True)
        else:
            st.info("No product impact data available.")

    # ---------------------------------------------------------
    # 5. Ranked Decision Prioritization Table
    # ---------------------------------------------------------
    st.markdown('<div class="section-header">📋 4. Prioritized Recommendations Table & Decision Drivers</div>', unsafe_allow_html=True)
    st.markdown(
        "Candidate recommendations ranked by **Recommendation Score** and potential distance savings. "
        "The **Key Decision Drivers** column transparently breaks down why each order ranks near the top. "
        "*Note: Recommendation Score (0–100) is a heuristic decision-support rank, not a calibrated statistical probability.*"
    )

    p_table = impact_results["prioritized_df"]

    display_cols = [
        "Order ID",
        "City",
        "Product Name",
        "Division",
        "Current_Factory",
        "Best_Alternative_Factory" if "Best_Alternative_Factory" in p_table.columns else "Selected Destination Factory",
        "Current_Factory_Distance_KM" if "Current_Factory_Distance_KM" in p_table.columns else "Current Distance (KM)",
        "Best_Alternative_Distance_KM" if "Best_Alternative_Distance_KM" in p_table.columns else "Scenario Distance (KM)",
        "Alternative_Distance_Saving_KM" if "Alternative_Distance_Saving_KM" in p_table.columns else "Distance Change (KM)",
        "Alternative_Distance_Saving_Pct" if "Alternative_Distance_Saving_Pct" in p_table.columns else "Distance Change (%)",
        "Recommendation_Score" if "Recommendation_Score" in p_table.columns else "Distance Change (KM)",
        "Operational_Compatibility",
        "Illustrative_Cost_Difference_$",
        "Key_Decision_Drivers",
    ]
    # Filter to existing columns only
    valid_disp_cols = [c for c in display_cols if c in p_table.columns]
    table_subset = p_table[valid_disp_cols].head(150).copy()

    st.dataframe(
        table_subset.style.format(
            {
                "Current_Factory_Distance_KM": "{:,.1f} km",
                "Best_Alternative_Distance_KM": "{:,.1f} km",
                "Current Distance (KM)": "{:,.1f} km",
                "Scenario Distance (KM)": "{:,.1f} km",
                "Alternative_Distance_Saving_KM": "{:+,.1f} km",
                "Distance Change (KM)": "{:+,.1f} km",
                "Alternative_Distance_Saving_Pct": "{:+.1f}%",
                "Distance Change (%)": "{:+.1f}%",
                "Recommendation_Score": "{:.1f}",
                "Illustrative_Cost_Difference_$": "${:+,.2f}",
            },
            na_rep="N/A",
        ),
        use_container_width=True,
        height=380,
    )

    # Download Button (Full scenario dataset)
    csv_bytes = p_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download Full Scenario Decision Intelligence Table as CSV ({len(p_table):,} Records)",
        data=csv_bytes,
        file_name=f"decision_intelligence_scenario_{cost_per_km:.2f}perkm.csv",
        mime="text/csv",
        help="Export the complete prioritized scenario table with driver explanations and compatibility tags.",
    )

    # ---------------------------------------------------------
    # 6. Operational Risk, Assumptions & Human Reviewer Roadmap
    # ---------------------------------------------------------
    st.markdown('<div class="section-header">🧭 5. Operational Risks, Assumptions & Action Roadmap for Human Reviewers</div>', unsafe_allow_html=True)

    risk_col1, risk_col2 = st.columns(2)

    with risk_col1:
        st.markdown(
            """
            **⚠️ Key Operational Constraints & Unknowns:**
            - **Plant Equipment & Allergen Controls:** Lot's O' Nuts processes tree nuts/peanuts; transferring chocolate production lines requires thorough allergen cleaning validation.
            - **Facility Capacity Ceilings:** Plant production capacities (batches/day, shift schedules) are **not modeled** in the dataset and must be confirmed prior to shifting physical orders.
            - **Circuity Multiplier:** Real highway road routes are typically 1.15–1.30× longer than straight-line Haversine estimates due to road infrastructure.
            - **Minimum Order Quantity (MOQ):** Confectionery batch minimums may restrict single-order reallocations without combining production runs.
            """
        )

    with risk_col2:
        st.markdown(
            """
            **📋 Recommended Next Actions for Logistics Managers:**
            1. **Verify High-Priority Candidates (216 orders):** Conduct plant-by-plant review of immediate recommendations (Score ≥ 70).
            2. **Consult Freight Carrier Contracts:** Compare illustrative $/km proxies with actual carrier lane tariffs (LTL vs FTL rates).
            3. **Review Incompatible / Cross-Division Moves:** Flag orders where chocolate products are routed to non-chocolate facilities.
            4. **Pilot Small Regional Batch:** Conduct a trial relocation of 50–100 orders to assess real-world carrier transit performance.
            """
        )

    # ---------------------------------------------------------
    # 7. Disclaimers & Methodology Safeguards
    # ---------------------------------------------------------
    _render_business_disclaimer()


def _render_business_disclaimer():
    """Renders business limitations and methodology boundaries."""
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Business Modeling Assumptions, Freight Proxies & Decision Safeguards:</strong><br>
            • <strong>Illustrative Freight Proxy Only:</strong> The cost calculations presented in this module use a user-defined 
              flat cost-per-kilometer multiplier ($/km) applied to straight-line Haversine distance. They serve strictly as 
              an illustrative proxy for comparative scenario analysis and do <em>not</em> represent contractual carrier tariffs, 
              fuel surcharges, accessorial fees, or guaranteed dollar savings.<br>
            • <strong>Heuristic Prioritization:</strong> The Recommendation Score (0–100) is a weighted heuristic decision-support rank 
              combining distance reduction, facility burden relief, product margin, and volume. It is <em>not</em> a calibrated 
              statistical probability of operational success.<br>
            • <strong>Manufacturing & Equipment Feasibility:</strong> The system does not assume production feasibility where factory 
              specializations mismatch product divisions. Actual confectionery manufacturing requires validation of equipment 
              compatibility, batch sizes, clean-in-place (CIP) protocols, and shift staffing.<br>
            • <strong>Decision Support:</strong> This system provides strategic intelligence to guide human decision-makers and 
              does not automate production dispatch or contractual carrier assignments.
        </div>
        """,
        unsafe_allow_html=True,
    )
