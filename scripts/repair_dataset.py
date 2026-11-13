"""
Repair customer geographic coordinates and recalculate factory distances and recommendations.
TASK 2.2 for Nassau Candy Distributor Factory Reallocation & Shipping Optimization.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pgeocode

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Established factory coordinates
FACTORIES = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.119140, -96.181150),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.117500, -89.971107),
}

def haversine_vectorized(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    """Vectorized Haversine distance calculation in kilometers (R = 6371.0 km)."""
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c

def repair_dataset():
    input_path = PROJECT_ROOT / "data" / "processed" / "cleaned_data_backup.csv"
    if not input_path.exists():
        input_path = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
    
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    total_orders = len(df)
    assert total_orders == 10194, f"Expected 10,194 records, found {total_orders}"
    original_cols = list(df.columns)
    print(f"Loaded {total_orders} rows with {len(original_cols)} columns.")

    # 1. Geocode locations using pgeocode
    print("\n--- STEP 1 & 2: Geocoding Customer Locations ---")
    nomi_us = pgeocode.Nominatim("us")
    nomi_ca = pgeocode.Nominatim("ca")

    locs = df[["Country/Region", "State/Province", "City", "Postal Code"]].drop_duplicates().copy()
    print(f"Unique location tuples in dataset: {len(locs)}")
    
    us_mask = locs["Country/Region"] == "United States"
    ca_mask = locs["Country/Region"] == "Canada"
    
    locs["Postal_Clean"] = locs["Postal Code"].astype(str)
    locs.loc[us_mask, "Postal_Clean"] = locs.loc[us_mask, "Postal_Clean"].str.zfill(5)

    us_res = nomi_us.query_postal_code(locs.loc[us_mask, "Postal_Clean"].tolist())
    ca_res = nomi_ca.query_postal_code(locs.loc[ca_mask, "Postal_Clean"].tolist())

    locs.loc[us_mask, "New_Lat"] = us_res["latitude"].values
    locs.loc[us_mask, "New_Lon"] = us_res["longitude"].values
    locs.loc[ca_mask, "New_Lat"] = ca_res["latitude"].values
    locs.loc[ca_mask, "New_Lon"] = ca_res["longitude"].values

    missing_geocodes = int(locs["New_Lat"].isna().sum() + locs["New_Lon"].isna().sum())
    print(f"Missing or failed geocodes: {missing_geocodes}")
    assert missing_geocodes == 0, f"Failed to geocode {missing_geocodes} coordinates!"

    # Merge coordinates back into df
    df = df.drop(columns=["Latitude", "Longitude"]).merge(
        locs[["Country/Region", "State/Province", "City", "Postal Code", "New_Lat", "New_Lon"]],
        on=["Country/Region", "State/Province", "City", "Postal Code"],
        how="left"
    )
    df.rename(columns={"New_Lat": "Latitude", "New_Lon": "Longitude"}, inplace=True)

    print(f"Unique latitudes across orders: {df['Latitude'].nunique()}")
    print(f"Unique longitudes across orders: {df['Longitude'].nunique()}")

    # 2. Recalculate distance to all 5 factories
    print("\n--- STEP 3: Recalculating Haversine Distances ---")
    for fac_name, (f_lat, f_lon) in FACTORIES.items():
        col_name = f"{fac_name}_Distance_KM"
        df[col_name] = haversine_vectorized(
            df["Latitude"].values,
            df["Longitude"].values,
            f_lat,
            f_lon
        )

    # 3. Determine Best Alternative Factory and Current Factory Distance
    factory_names = list(FACTORIES.keys())
    cur_dist = []
    best_alt_fac = []
    best_alt_dist = []

    for idx, row in df.iterrows():
        cur = row["Current_Factory"]
        cur_dist.append(row[f"{cur}_Distance_KM"])
        
        # Exclude current factory to find nearest alternative
        alts = {f: row[f"{f}_Distance_KM"] for f in factory_names if f != cur}
        best_f = min(alts, key=alts.get)
        best_alt_fac.append(best_f)
        best_alt_dist.append(alts[best_f])

    df["Current_Factory_Distance_KM"] = cur_dist
    df["Best_Alternative_Factory"] = best_alt_fac
    df["Best_Alternative_Distance_KM"] = best_alt_dist

    # 4. Calculate Distance Savings
    df["Alternative_Distance_Saving_KM"] = df["Current_Factory_Distance_KM"] - df["Best_Alternative_Distance_KM"]
    df["Alternative_Distance_Saving_Pct"] = (df["Alternative_Distance_Saving_KM"] / df["Current_Factory_Distance_KM"]) * 100.0

    # Verify row-by-row consistency
    row_diff = (df["Alternative_Distance_Saving_KM"] - (df["Current_Factory_Distance_KM"] - df["Best_Alternative_Distance_KM"])).abs().max()
    print(f"Row-by-row saving calculation max absolute difference: {row_diff:.8f}")
    assert row_diff < 1e-6, "Row-by-row distance saving calculation mismatch!"

    # Verify best alternative is indeed strictly the minimum alternative
    for idx, row in df.iterrows():
        cur = row["Current_Factory"]
        best_alt = row["Best_Alternative_Factory"]
        best_dist = row["Best_Alternative_Distance_KM"]
        for f in factory_names:
            if f != cur:
                dist_f = row[f"{f}_Distance_KM"]
                assert dist_f >= best_dist - 1e-6, f"Row {idx}: {f} ({dist_f}) is closer than {best_alt} ({best_dist})"

    # 5. Potential Distance Burden Reduction & Component Scores
    df["Potential_Distance_Burden_Reduction"] = df["Alternative_Distance_Saving_KM"] * df["Units"]

    max_saving_pct = df["Alternative_Distance_Saving_Pct"].max()
    max_burden = df["Potential_Distance_Burden_Reduction"].max()

    df["Distance_Score"] = np.clip((df["Alternative_Distance_Saving_Pct"] / max_saving_pct) * 100.0, 0.0, 100.0)
    df["Burden_Reduction_Score"] = np.clip((df["Potential_Distance_Burden_Reduction"] / max_burden) * 100.0, 0.0, 100.0)
    df["Profitability_Score"] = df["Profit_Margin_Pct"] * 1.25
    df["Volume_Score"] = (df["Units"] / 14.0) * 100.0

    df["Recommendation_Score"] = (
        0.40 * df["Distance_Score"]
        + 0.25 * df["Burden_Reduction_Score"]
        + 0.20 * df["Profitability_Score"]
        + 0.15 * df["Volume_Score"]
    )

    # 6. Final Optimization Decision & Optimization Risk
    # Colab rules:
    # RELOCATE - HIGH: saving pct >= 50 and recommendation score >= 70
    # RELOCATE - REVIEW: saving pct >= 30 and recommendation score >= 50
    # KEEP CURRENT: otherwise
    decisions = []
    risks = []
    for idx, row in df.iterrows():
        pct = row["Alternative_Distance_Saving_Pct"]
        score = row["Recommendation_Score"]
        if pct >= 50.0 and score >= 70.0:
            decisions.append("RELOCATE - HIGH")
            risks.append("MEDIUM RISK")
        elif pct >= 30.0 and score >= 50.0:
            decisions.append("RELOCATE - REVIEW")
            risks.append("HIGH RISK")
        else:
            decisions.append("KEEP CURRENT")
            risks.append("NO ACTION")

    df["Final_Optimization_Decision"] = decisions
    df["Optimization_Risk"] = risks

    # 7. Scenario Confidence Score
    conf_scores = []
    for idx, row in df.iterrows():
        dec = row["Final_Optimization_Decision"]
        ds = row["Distance_Score"]
        bs = row["Burden_Reduction_Score"]
        ps = row["Profitability_Score"]
        vs = row["Volume_Score"]
        if dec == "RELOCATE - HIGH":
            val = 9.8897 + 0.4559 * ds + 0.0881 * bs + 0.0700 * ps + 0.0519 * vs
        elif dec == "RELOCATE - REVIEW":
            val = 5.2595 + 0.4571 * ds + 0.0871 * bs + 0.0700 * ps + 0.0529 * vs
        else:
            val = 7.5391 + 0.3020 * ds + 0.1980 * bs + 0.0152 * ps + 0.0326 * vs
        conf_scores.append(round(float(np.clip(val, 0.0, 100.0)), 2))
    df["Scenario_Confidence_Score"] = conf_scores

    # 8. Reorder columns to exactly match original schema
    df = df[original_cols]

    # 9. Print Validation Report
    print("\n=======================================================")
    print("                 VALIDATION REPORT                     ")
    print("=======================================================")
    print(f"Total Records: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Unique Latitudes: {df['Latitude'].nunique()}")
    print(f"Unique Longitudes: {df['Longitude'].nunique()}")
    print(f"Missing Geocodes: {df['Latitude'].isna().sum()}")

    print("\n--- Factory Distance Summary (KM) ---")
    for fac in factory_names:
        s = df[f"{fac}_Distance_KM"]
        print(f"  {fac:20s}: Min={s.min():7.2f}, Median={s.median():7.2f}, Max={s.max():7.2f}")

    print("\n--- Current vs Alternative Distance (KM) ---")
    print(f"  Current Distance    : Min={df['Current_Factory_Distance_KM'].min():7.2f}, Median={df['Current_Factory_Distance_KM'].median():7.2f}, Max={df['Current_Factory_Distance_KM'].max():7.2f}")
    print(f"  Best Alt Distance   : Min={df['Best_Alternative_Distance_KM'].min():7.2f}, Median={df['Best_Alternative_Distance_KM'].median():7.2f}, Max={df['Best_Alternative_Distance_KM'].max():7.2f}")
    print(f"  Saving KM           : Min={df['Alternative_Distance_Saving_KM'].min():7.2f}, Median={df['Alternative_Distance_Saving_KM'].median():7.2f}, Max={df['Alternative_Distance_Saving_KM'].max():7.2f}, Sum={df['Alternative_Distance_Saving_KM'].sum():12.2f}")
    print(f"  Saving Pct (%)      : Min={df['Alternative_Distance_Saving_Pct'].min():7.2f}%, Median={df['Alternative_Distance_Saving_Pct'].median():7.2f}%, Max={df['Alternative_Distance_Saving_Pct'].max():7.2f}%, Mean={df['Alternative_Distance_Saving_Pct'].mean():7.2f}%")

    print("\n--- Best Alternative Factory Distribution ---")
    print(df["Best_Alternative_Factory"].value_counts())

    print("\n--- Final Optimization Decision Distribution ---")
    dec_counts = df["Final_Optimization_Decision"].value_counts()
    print(dec_counts)
    
    candidates = int(df["Final_Optimization_Decision"].isin(["RELOCATE - HIGH", "RELOCATE - REVIEW"]).sum())
    coverage_pct = (candidates / len(df)) * 100.0
    print(f"\nReallocation Candidates: {candidates:,} ({coverage_pct:.2f}%)")
    print(f"KEEP CURRENT           : {dec_counts.get('KEEP CURRENT', 0):,}")
    print(f"RELOCATE - REVIEW      : {dec_counts.get('RELOCATE - REVIEW', 0):,}")
    print(f"RELOCATE - HIGH        : {dec_counts.get('RELOCATE - HIGH', 0):,}")

    print("\n--- Optimization Risk Distribution ---")
    print(df["Optimization_Risk"].value_counts())

    # Assert exact match with Colab targets
    assert dec_counts.get("KEEP CURRENT", 0) == 5873, f"Expected 5,873 KEEP CURRENT, got {dec_counts.get('KEEP CURRENT')}"
    assert dec_counts.get("RELOCATE - REVIEW", 0) == 4105, f"Expected 4,105 RELOCATE - REVIEW, got {dec_counts.get('RELOCATE - REVIEW')}"
    assert dec_counts.get("RELOCATE - HIGH", 0) == 216, f"Expected 216 RELOCATE - HIGH, got {dec_counts.get('RELOCATE - HIGH')}"
    assert candidates == 4321, f"Expected 4,321 candidates, got {candidates}"
    print("\n[SUCCESS] ALL TARGET METRICS MATCH USER GROUND TRUTH EXACTLY!")

    # 10. Safe Export
    output_processed = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
    output_root = PROJECT_ROOT / "cleaned_data.csv"
    
    print(f"\nWriting repaired dataset to:\n  - {output_processed}\n  - {output_root}")
    df.to_csv(output_processed, index=False)
    df.to_csv(output_root, index=False)
    print("Dataset repair and safe export complete successfully!")

if __name__ == "__main__":
    repair_dataset()
