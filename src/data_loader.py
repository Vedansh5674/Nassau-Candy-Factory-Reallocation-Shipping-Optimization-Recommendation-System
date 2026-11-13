"""
Factory Reallocation & Shipping Optimization Recommendation System
Shared Data Loading and Validation Module for Nassau Candy Distributor.

This module loads and validates the processed dataset ('data/processed/cleaned_data.csv').
It enforces schema integrity, checks required columns, data types, value boundaries,
and provides caching for Streamlit alongside standalone execution capabilities.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

# Optional Streamlit caching decorator support
def get_cached_loader(func):
    """Wraps func with @st.cache_data if running in a Streamlit server runtime."""
    try:
        import streamlit as st
        from streamlit.runtime import exists as st_runtime_exists
        if st_runtime_exists():
            return st.cache_data(show_spinner=False)(func)
    except Exception:
        pass
    return func

# Standard Project Paths
DEFAULT_PROCESSED_PATH = Path("data") / "processed" / "cleaned_data.csv"
ROOT_FALLBACK_PATH = Path("cleaned_data.csv")

# Categorized Column Definitions (42 columns in total)
ORDER_INFO_COLUMNS = [
    "Row ID",
    "Order ID",
    "Order Date",
    "Ship Date",
    "Ship Mode",
    "Customer ID",
]

GEO_COLUMNS = [
    "Country/Region",
    "City",
    "State/Province",
    "Postal Code",
    "Division",
    "Region",
    "Latitude",
    "Longitude",
]

PRODUCT_FINANCIAL_COLUMNS = [
    "Product ID",
    "Product Name",
    "Sales",
    "Units",
    "Gross Profit",
    "Cost",
    "Lead_Time",
    "Profit_Margin_Pct",
]

FACTORY_DISTANCE_COLUMNS = [
    "Current_Factory",
    "Lot's O' Nuts_Distance_KM",
    "Wicked Choccy's_Distance_KM",
    "Sugar Shack_Distance_KM",
    "Secret Factory_Distance_KM",
    "The Other Factory_Distance_KM",
    "Current_Factory_Distance_KM",
    "Best_Alternative_Factory",
    "Best_Alternative_Distance_KM",
    "Alternative_Distance_Saving_KM",
    "Alternative_Distance_Saving_Pct",
    "Potential_Distance_Burden_Reduction",
]

RECOMMENDATION_SCORE_COLUMNS = [
    "Distance_Score",
    "Burden_Reduction_Score",
    "Profitability_Score",
    "Volume_Score",
    "Recommendation_Score",
]

DECISION_RISK_COLUMNS = [
    "Final_Optimization_Decision",
    "Optimization_Risk",
    "Scenario_Confidence_Score",
]

# Complete set of 42 expected columns
ALL_EXPECTED_COLUMNS = (
    ORDER_INFO_COLUMNS
    + GEO_COLUMNS
    + PRODUCT_FINANCIAL_COLUMNS
    + FACTORY_DISTANCE_COLUMNS
    + RECOMMENDATION_SCORE_COLUMNS
    + DECISION_RISK_COLUMNS
)

# Allowed categorical values for validation
VALID_DECISIONS = {"RELOCATE - REVIEW", "RELOCATE - HIGH", "KEEP CURRENT"}
VALID_RISK_LEVELS = {"HIGH RISK", "MEDIUM RISK", "NO ACTION"}


def resolve_data_path(custom_path: Optional[str] = None) -> Path:
    """
    Resolves the dataset file path checking custom, default, and root locations.
    """
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Custom dataset path does not exist: {custom_path}")

    if DEFAULT_PROCESSED_PATH.exists():
        return DEFAULT_PROCESSED_PATH

    if ROOT_FALLBACK_PATH.exists():
        return ROOT_FALLBACK_PATH

    raise FileNotFoundError(
        f"Cleaned dataset not found. Checked:\n"
        f"  1. {DEFAULT_PROCESSED_PATH.resolve()}\n"
        f"  2. {ROOT_FALLBACK_PATH.resolve()}"
    )


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates dataset integrity against the required schema.

    Returns:
        dict: Diagnostic dictionary containing 'is_valid', 'missing_columns',
              'row_count', 'column_count', 'null_counts', and 'warnings'.
    """
    results: Dict[str, Any] = {
        "is_valid": True,
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_columns": [],
        "null_counts": {},
        "warnings": [],
        "errors": [],
    }

    # 1. Check for missing columns
    missing = [col for col in ALL_EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        results["is_valid"] = False
        results["missing_columns"] = missing
        results["errors"].append(f"Missing {len(missing)} required column(s): {missing}")

    # 2. Check for null values
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0].to_dict()
    if cols_with_nulls:
        results["null_counts"] = cols_with_nulls
        results["warnings"].append(f"Columns containing nulls: {cols_with_nulls}")

    # 3. Categorical integrity checks
    if "Final_Optimization_Decision" in df.columns:
        unexpected_decisions = set(df["Final_Optimization_Decision"].unique()) - VALID_DECISIONS
        if unexpected_decisions:
            results["warnings"].append(f"Unexpected values in Final_Optimization_Decision: {unexpected_decisions}")

    if "Optimization_Risk" in df.columns:
        unexpected_risks = set(df["Optimization_Risk"].unique()) - VALID_RISK_LEVELS
        if unexpected_risks:
            results["warnings"].append(f"Unexpected values in Optimization_Risk: {unexpected_risks}")

    # 4. Numerical range sanity checks
    score_cols = [c for c in RECOMMENDATION_SCORE_COLUMNS if c in df.columns]
    for col in score_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_min < 0 or col_max > 100.01:
            results["warnings"].append(f"Column '{col}' has values outside [0, 100]: min={col_min}, max={col_max}")

    return results


@get_cached_loader
def load_data(
    filepath: Optional[str] = None,
    parse_dates: bool = True,
    validate: bool = True
) -> pd.DataFrame:
    """
    Loads and validates the Nassau Candy factory reallocation dataset.

    Args:
        filepath: Optional custom file path. If None, resolves standard locations.
        parse_dates: Whether to parse 'Order Date' and 'Ship Date' into datetime format.
        validate: Whether to run schema validation upon loading.

    Returns:
        pd.DataFrame: Cleaned and validated dataset.
    """
    resolved_path = resolve_data_path(filepath)
    df = pd.read_csv(resolved_path)

    if validate:
        validation_report = validate_dataset(df)
        if not validation_report["is_valid"]:
            error_msg = "; ".join(validation_report["errors"])
            raise ValueError(f"Dataset validation failed: {error_msg}")

    if parse_dates:
        for date_col in ["Order Date", "Ship Date"]:
            if date_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    return df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates high-level statistical and business summary metrics
    reusing existing analytical results from the Colab pipeline.
    """
    cur_dist_sum = float(df["Current_Factory_Distance_KM"].sum()) if "Current_Factory_Distance_KM" in df.columns else 0.0
    saved_dist_sum = float(df["Alternative_Distance_Saving_KM"].sum()) if "Alternative_Distance_Saving_KM" in df.columns else 0.0
    agg_saving_pct = (saved_dist_sum / cur_dist_sum * 100.0) if cur_dist_sum > 0 else 0.0

    cand_df = (
        df[df["Final_Optimization_Decision"].isin(["RELOCATE - HIGH", "RELOCATE - REVIEW"])]
        if "Final_Optimization_Decision" in df.columns
        else pd.DataFrame()
    )
    cand_orders = len(cand_df)
    cand_avg_saving_pct = (
        float(cand_df["Alternative_Distance_Saving_Pct"].mean())
        if not cand_df.empty and "Alternative_Distance_Saving_Pct" in cand_df.columns
        else 0.0
    )
    cand_cur_dist = (
        float(cand_df["Current_Factory_Distance_KM"].sum())
        if not cand_df.empty and "Current_Factory_Distance_KM" in cand_df.columns
        else 0.0
    )
    cand_saved_dist = (
        float(cand_df["Alternative_Distance_Saving_KM"].sum())
        if not cand_df.empty and "Alternative_Distance_Saving_KM" in cand_df.columns
        else 0.0
    )
    cand_agg_saving_pct = (cand_saved_dist / cand_cur_dist * 100.0) if cand_cur_dist > 0 else 0.0

    summary = {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "total_sales": float(df["Sales"].sum()) if "Sales" in df.columns else 0.0,
        "total_gross_profit": float(df["Gross Profit"].sum()) if "Gross Profit" in df.columns else 0.0,
        "total_units": int(df["Units"].sum()) if "Units" in df.columns else 0,
        "avg_profit_margin_pct": float(df["Profit_Margin_Pct"].mean()) if "Profit_Margin_Pct" in df.columns else 0.0,
        "total_current_distance_km": cur_dist_sum,
        "total_distance_saved_km": saved_dist_sum,
        "avg_current_distance_km": float(df["Current_Factory_Distance_KM"].mean()) if "Current_Factory_Distance_KM" in df.columns else 0.0,
        "avg_alt_distance_km": float(df["Best_Alternative_Distance_KM"].mean()) if "Best_Alternative_Distance_KM" in df.columns else 0.0,
        "avg_distance_saving_km": float(df["Alternative_Distance_Saving_KM"].mean()) if "Alternative_Distance_Saving_KM" in df.columns else 0.0,
        "avg_distance_saving_pct": float(df["Alternative_Distance_Saving_Pct"].mean()) if "Alternative_Distance_Saving_Pct" in df.columns else 0.0,
        "aggregate_distance_saving_pct": agg_saving_pct,
        "candidate_orders": cand_orders,
        "candidate_avg_distance_saving_pct": cand_avg_saving_pct,
        "candidate_aggregate_distance_saving_pct": cand_agg_saving_pct,
        "decision_counts": df["Final_Optimization_Decision"].value_counts().to_dict() if "Final_Optimization_Decision" in df.columns else {},
        "risk_counts": df["Optimization_Risk"].value_counts().to_dict() if "Optimization_Risk" in df.columns else {},
        "current_factory_distribution": df["Current_Factory"].value_counts().to_dict() if "Current_Factory" in df.columns else {},
        "best_alt_factory_distribution": df["Best_Alternative_Factory"].value_counts().to_dict() if "Best_Alternative_Factory" in df.columns else {},
        "avg_recommendation_score": float(df["Recommendation_Score"].mean()) if "Recommendation_Score" in df.columns else 0.0,
        "avg_confidence_score": float(df["Scenario_Confidence_Score"].mean()) if "Scenario_Confidence_Score" in df.columns else 0.0,
    }
    return summary


if __name__ == "__main__":
    print("Executing standalone validation test of src/data_loader.py...")
    resolved = resolve_data_path()
    print(f"Resolved path: {resolved}")
    data = load_data()
    print(f"Data successfully loaded. Shape: {data.shape}")
    rep = validate_dataset(data)
    print(f"Validation passed: {rep['is_valid']}")
    summary = get_dataset_summary(data)
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
