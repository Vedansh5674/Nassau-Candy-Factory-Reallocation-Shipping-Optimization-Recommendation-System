"""
Task 6: Comprehensive End-to-End Validation and Business Logic Audit Regression Tests.
Verifies dataset integrity, KPI formulas, order-level vs aggregate percentages,
all 20 factory pairs, cross-module consistency, error handling, and business safeguards.
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data, validate_dataset, get_dataset_summary
from src.simulator import (
    FACTORY_METADATA,
    ALL_FACTORIES,
    DISTANCE_TOLERANCE_KM,
    get_factory_distance,
    validate_simulation_inputs,
    filter_simulation_orders,
    run_factory_simulation,
)
from src.geo_network import (
    validate_geo_coordinates,
    get_factory_summary,
    aggregate_customer_locations,
    build_assignment_comparison_map,
)


class TestTask6AuditValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Loads and validates the baseline dataset once for all audit tests."""
        cls.df = load_data(validate=True, parse_dates=True)
        cls.report = validate_dataset(cls.df)

    # ------------------------------------------------------------------
    # 1. Dataset Integrity & Baseline Verification
    # ------------------------------------------------------------------
    def test_01_dataset_baseline_and_integrity(self):
        """Verify row count, column count, duplicates, nulls, and exact recommendation baseline."""
        df = self.df
        self.assertTrue(self.report["is_valid"])
        self.assertEqual(len(df), 10194, "Dataset must contain exactly 10,194 records.")
        self.assertEqual(len(df.columns), 42, "Dataset must contain exactly 42 columns.")
        self.assertEqual(df.duplicated().sum(), 0, "Dataset must have 0 duplicate rows.")
        self.assertEqual(df["Row ID"].duplicated().sum(), 0, "Row ID must be strictly unique.")
        self.assertEqual(df.isnull().sum().sum(), 0, "Dataset must have 0 null values across all columns.")

        # Baseline recommendation counts
        dec_counts = df["Final_Optimization_Decision"].value_counts()
        self.assertEqual(dec_counts.get("KEEP CURRENT", 0), 5873)
        self.assertEqual(dec_counts.get("RELOCATE - REVIEW", 0), 4105)
        self.assertEqual(dec_counts.get("RELOCATE - HIGH", 0), 216)
        total_candidates = dec_counts.get("RELOCATE - REVIEW", 0) + dec_counts.get("RELOCATE - HIGH", 0)
        self.assertEqual(total_candidates, 4321)

        # Baseline risk counts
        risk_counts = df["Optimization_Risk"].value_counts()
        self.assertEqual(risk_counts.get("NO ACTION", 0), 5873)
        self.assertEqual(risk_counts.get("HIGH RISK", 0), 4105)
        self.assertEqual(risk_counts.get("MEDIUM RISK", 0), 216)

    def test_02_customer_coordinates_distribution(self):
        """Confirm customer coordinates are not collapsed to a national centroid."""
        df = self.df
        geo_diag = validate_geo_coordinates(df)
        self.assertFalse(geo_diag["has_single_centroid_error"])
        self.assertEqual(geo_diag["lat_nulls"], 0)
        self.assertEqual(geo_diag["lon_nulls"], 0)
        self.assertEqual(geo_diag["unique_locations"], 654)

        # Confirm non-trivial dispersion across geography
        self.assertGreater(df["Latitude"].std(), 4.0)
        self.assertGreater(df["Longitude"].std(), 15.0)

        # Confirm bounding coordinates match North America
        self.assertGreaterEqual(df["Latitude"].min(), 25.0)
        self.assertLessEqual(df["Latitude"].max(), 54.0)
        self.assertGreaterEqual(df["Longitude"].min(), -125.0)
        self.assertLessEqual(df["Longitude"].max(), -50.0)

    def test_03_factory_coordinates_and_distance_columns(self):
        """Confirm all 5 factory coordinates and verify that distance columns match Haversine recalculation."""
        df = self.df
        expected_meta = {
            "Lot's O' Nuts": {"lat": 32.881893, "lon": -111.768036, "col": "Lot's O' Nuts_Distance_KM"},
            "Wicked Choccy's": {"lat": 32.076176, "lon": -81.088371, "col": "Wicked Choccy's_Distance_KM"},
            "Sugar Shack": {"lat": 48.119140, "lon": -96.181150, "col": "Sugar Shack_Distance_KM"},
            "Secret Factory": {"lat": 41.446333, "lon": -90.565487, "col": "Secret Factory_Distance_KM"},
            "The Other Factory": {"lat": 35.117500, "lon": -89.971107, "col": "The Other Factory_Distance_KM"},
        }

        R = 6371.0
        for name, meta in expected_meta.items():
            self.assertIn(name, FACTORY_METADATA)
            self.assertAlmostEqual(FACTORY_METADATA[name]["latitude"], meta["lat"], places=5)
            self.assertAlmostEqual(FACTORY_METADATA[name]["longitude"], meta["lon"], places=5)
            self.assertEqual(FACTORY_METADATA[name]["distance_column"], meta["col"])
            self.assertIn(meta["col"], df.columns)

            # Recalculate Haversine distance from customer coordinates to factory
            phi1 = np.radians(df["Latitude"])
            phi2 = np.radians(meta["lat"])
            dphi = np.radians(meta["lat"] - df["Latitude"])
            dlambda = np.radians(meta["lon"] - df["Longitude"])
            a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
            recomputed = R * (2.0 * np.arcsin(np.sqrt(a)))

            diff = (df[meta["col"]] - recomputed).abs().max()
            self.assertLess(diff, 0.001, f"Factory distance column {meta['col']} differs from Haversine calculation.")

        # Confirm Current_Factory_Distance_KM matches Current_Factory's distance column
        for fac_name in ALL_FACTORIES:
            fac_orders = df[df["Current_Factory"] == fac_name]
            col_name = FACTORY_METADATA[fac_name]["distance_column"]
            diff = (fac_orders["Current_Factory_Distance_KM"] - fac_orders[col_name]).abs().max()
            self.assertLess(diff, 1e-4)

        # Confirm Best_Alternative_Factory is strictly the closest facility among the other 4
        for idx, row in df.sample(n=200, random_state=42).iterrows():
            cur_f = row["Current_Factory"]
            other_facs = [f for f in ALL_FACTORIES if f != cur_f]
            dists = {f: row[FACTORY_METADATA[f]["distance_column"]] for f in other_facs}
            min_fac = min(dists, key=dists.get)
            self.assertEqual(min_fac, row["Best_Alternative_Factory"])

    # ------------------------------------------------------------------
    # 2. Audit Dashboard Metrics & Distance Saving Percentages
    # ------------------------------------------------------------------
    def test_04_kpi_formulas_and_saving_percentage_audit(self):
        """
        Audit KPI formulas: investigate order-level mean (-18.1%) vs candidate-only (77.8%)
        vs aggregate distance reduction (+46.1%), documenting populations, numerators, and denominators.
        """
        df = self.df
        total_orders = len(df)
        total_sales = float(df["Sales"].sum())
        total_profit = float(df["Gross Profit"].sum())
        total_units = int(df["Units"].sum())

        self.assertEqual(total_orders, 10194)
        self.assertAlmostEqual(total_sales, 141783.63, places=2)
        self.assertAlmostEqual(total_profit, 93442.80, places=2)
        self.assertEqual(total_units, 38654)

        # Candidate breakdown
        cand_df = df[df["Final_Optimization_Decision"].isin(["RELOCATE - HIGH", "RELOCATE - REVIEW"])]
        keep_df = df[df["Final_Optimization_Decision"] == "KEEP CURRENT"]
        self.assertEqual(len(cand_df), 4321)
        self.assertEqual(len(keep_df), 5873)

        # Distance totals
        total_current_km = float(df["Current_Factory_Distance_KM"].sum())
        total_alt_km = float(df["Best_Alternative_Distance_KM"].sum())
        total_saved_km = float(df["Alternative_Distance_Saving_KM"].sum())

        self.assertAlmostEqual(total_current_km, 20480868.06, places=1)
        self.assertAlmostEqual(total_alt_km, 11034386.34, places=1)
        self.assertAlmostEqual(total_saved_km, 9446481.72, places=1)
        self.assertAlmostEqual(total_saved_km, total_current_km - total_alt_km, places=1)

        # 1. Order-level mean saving percentage across all orders (displays as -18.1%)
        # Formula: (1 / N) * sum((Current - Alt) / Current * 100)
        mean_order_saving_pct = float(df["Alternative_Distance_Saving_Pct"].mean())
        self.assertAlmostEqual(mean_order_saving_pct, -18.058, places=2)

        # 2. Candidate-only order-level mean saving percentage (displays as 77.8%)
        # Formula: (1 / N_cand) * sum_cand((Current - Alt) / Current * 100)
        cand_mean_saving_pct = float(cand_df["Alternative_Distance_Saving_Pct"].mean())
        self.assertAlmostEqual(cand_mean_saving_pct, 77.754, places=2)

        # 3. Aggregate distance reduction across all orders (displays +46.1%)
        # Formula: (Total_Saved_KM / Total_Current_KM) * 100
        agg_reduction_pct_all = (total_saved_km / total_current_km) * 100.0
        self.assertAlmostEqual(agg_reduction_pct_all, 46.123, places=2)

        # 4. Aggregate distance reduction for candidates (displays 76.8%)
        cand_saved_km = float(cand_df["Alternative_Distance_Saving_KM"].sum())
        cand_current_km = float(cand_df["Current_Factory_Distance_KM"].sum())
        cand_agg_reduction_pct = (cand_saved_km / cand_current_km) * 100.0
        self.assertAlmostEqual(cand_agg_reduction_pct, 76.803, places=2)

        # Verify that negative distance saving means additional geographic distance
        keep_negative = keep_df[keep_df["Alternative_Distance_Saving_KM"] < 0]
        self.assertEqual(len(keep_negative), 3227)
        for _, row in keep_negative.head(20).iterrows():
            self.assertGreater(row["Best_Alternative_Distance_KM"], row["Current_Factory_Distance_KM"])
            self.assertLess(row["Alternative_Distance_Saving_KM"], 0)
            self.assertLess(row["Alternative_Distance_Saving_Pct"], 0)

        # Test enhanced get_dataset_summary returns these metrics
        summary = get_dataset_summary(df)
        self.assertAlmostEqual(summary["avg_distance_saving_pct"], -18.058, places=2)
        self.assertAlmostEqual(summary["aggregate_distance_saving_pct"], 46.123, places=2)
        self.assertAlmostEqual(summary["candidate_avg_distance_saving_pct"], 77.754, places=2)
        self.assertAlmostEqual(summary["candidate_aggregate_distance_saving_pct"], 76.803, places=2)
        self.assertEqual(summary["candidate_orders"], 4321)

    # ------------------------------------------------------------------
    # 3. Simulator & Geographic Network Audits Across All 20 Factory Pairs
    # ------------------------------------------------------------------
    def test_05_all_twenty_factory_pairs_simulation(self):
        """Test all 20 distinct source/destination factory combinations across both modules."""
        df = self.df
        tested_pairs = 0

        for src in ALL_FACTORIES:
            for dst in ALL_FACTORIES:
                if src == dst:
                    # Confirm identical source and destination is rejected
                    valid, err = validate_simulation_inputs(df, src, dst)
                    self.assertFalse(valid)
                    self.assertIn("cannot be identical", err)
                    continue

                tested_pairs += 1
                res = run_factory_simulation(df, source_factory=src, destination_factory=dst)

                # Confirm current distance uses source column and scenario distance uses destination column
                src_col = FACTORY_METADATA[src]["distance_column"]
                dst_col = FACTORY_METADATA[dst]["distance_column"]
                comp_df = res["comparison_df"]
                src_df = df[df["Current_Factory"] == src].reset_index(drop=True)

                self.assertEqual(res["order_count"], len(src_df))
                self.assertAlmostEqual(
                    float((comp_df["Current Distance (KM)"] - src_df[src_col]).abs().max()),
                    0.0,
                    places=4,
                )
                self.assertAlmostEqual(
                    float((comp_df["Scenario Distance (KM)"] - src_df[dst_col]).abs().max()),
                    0.0,
                    places=4,
                )

                # Confirm decreased, increased, unchanged sum exactly to order count
                dec = res["orders_decreasing"]
                inc = res["orders_increasing"]
                unch = res["orders_unchanged"]
                self.assertEqual(dec + inc + unch, res["order_count"])

                # Confirm 0.001 km tolerance
                delta = comp_df["Current Distance (KM)"] - comp_df["Scenario Distance (KM)"]
                self.assertEqual(int((delta > DISTANCE_TOLERANCE_KM).sum()), dec)
                self.assertEqual(int((delta < -DISTANCE_TOLERANCE_KM).sum()), inc)
                self.assertEqual(int((delta.abs() <= DISTANCE_TOLERANCE_KM).sum()), unch)

                # Confirm sign convention
                saved = res["total_distance_saved_km"]
                cur_total = res["current_total_distance_km"]
                scen_total = res["scenario_total_distance_km"]
                self.assertAlmostEqual(saved, cur_total - scen_total, places=2)

                # Confirm aggregate percentage reduction formula
                expected_pct = (saved / cur_total) * 100.0 if cur_total > 0 else 0.0
                self.assertAlmostEqual(res["pct_reduction_total"], expected_pct, places=3)

        self.assertEqual(tested_pairs, 20)

    # ------------------------------------------------------------------
    # 4. Cross-Module Consistency (Simulator vs Geographic Network)
    # ------------------------------------------------------------------
    def test_06_cross_module_consistency(self):
        """Compare the same factory pair and filter settings in simulator and Geographic Network."""
        df = self.df
        test_pairs = [
            ("Lot's O' Nuts", "Wicked Choccy's"),
            ("Sugar Shack", "Secret Factory"),
            ("The Other Factory", "Lot's O' Nuts"),
        ]

        for src, dst in test_pairs:
            # 1. Full scope
            sim_full = run_factory_simulation(df, src, dst, candidates_only=False)
            src_orders = df[df["Current_Factory"] == src]
            src_col = FACTORY_METADATA[src]["distance_column"]
            dst_col = FACTORY_METADATA[dst]["distance_column"]

            delta_km = src_orders[src_col] - src_orders[dst_col]
            dec_expected = int((delta_km > DISTANCE_TOLERANCE_KM).sum())
            inc_expected = int((delta_km < -DISTANCE_TOLERANCE_KM).sum())
            unch_expected = int((delta_km.abs() <= DISTANCE_TOLERANCE_KM).sum())

            self.assertEqual(sim_full["orders_decreasing"], dec_expected)
            self.assertEqual(sim_full["orders_increasing"], inc_expected)
            self.assertEqual(sim_full["orders_unchanged"], unch_expected)

            # 2. Candidates only
            sim_cand = run_factory_simulation(df, src, dst, candidates_only=True)
            cand_orders = src_orders[src_orders["Final_Optimization_Decision"].isin(["RELOCATE - HIGH", "RELOCATE - REVIEW"])]
            delta_cand = cand_orders[src_col] - cand_orders[dst_col]
            dec_cand_expected = int((delta_cand > DISTANCE_TOLERANCE_KM).sum())

            self.assertEqual(sim_cand["orders_decreasing"], dec_cand_expected)
            self.assertEqual(sim_cand["order_count"], len(cand_orders))

    # ------------------------------------------------------------------
    # 5. Edge Cases: Zero Denominator, Empty Filters, Invalid Inputs
    # ------------------------------------------------------------------
    def test_07_edge_cases_and_error_handling(self):
        """Verify zero-distance denominators, empty filters, missing columns, and invalid selections."""
        df = self.df

        # Edge case: zero distance denominator protection
        df_zero = df.copy()
        df_zero.loc[0, FACTORY_METADATA["Lot's O' Nuts"]["distance_column"]] = 0.0
        res_zero = run_factory_simulation(df_zero, "Lot's O' Nuts", "Sugar Shack")
        self.assertFalse(res_zero["comparison_df"]["Distance Change (%)"].isnull().any())
        self.assertFalse(np.isinf(res_zero["comparison_df"]["Distance Change (%)"]).any())

        # Edge case: empty filter
        res_empty = run_factory_simulation(df, "Sugar Shack", "Lot's O' Nuts", min_saving_pct=99.9)
        self.assertTrue(res_empty["is_empty"])
        self.assertEqual(res_empty["order_count"], 0)
        self.assertEqual(res_empty["current_total_distance_km"], 0.0)
        self.assertEqual(res_empty["orders_decreasing"], 0)
        self.assertEqual(res_empty["orders_increasing"], 0)
        self.assertEqual(res_empty["orders_unchanged"], 0)

        # Edge case: invalid factory selection
        valid, err = validate_simulation_inputs(df, "Imaginary Factory", "Lot's O' Nuts")
        self.assertFalse(valid)
        self.assertIn("not recognized", err)

        # Edge case: missing column
        df_no_col = df.drop(columns=[FACTORY_METADATA["The Other Factory"]["distance_column"]])
        valid, err = validate_simulation_inputs(df_no_col, "Lot's O' Nuts", "The Other Factory")
        self.assertFalse(valid)
        self.assertIn("missing from dataset", err)

    # ------------------------------------------------------------------
    # 6. CSV Export Column Schemas
    # ------------------------------------------------------------------
    def test_08_csv_export_schemas(self):
        """Verify CSV exports in Dashboard and Simulator contain expected records and columns."""
        df = self.df

        # Dashboard export columns
        expected_dashboard_cols = [
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
        for col in expected_dashboard_cols:
            self.assertIn(col, df.columns)

        # Simulator export columns
        sim_res = run_factory_simulation(df, "Lot's O' Nuts", "Sugar Shack")
        expected_sim_cols = [
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
        self.assertEqual(list(sim_res["comparison_df"].columns), expected_sim_cols)


if __name__ == "__main__":
    unittest.main()
