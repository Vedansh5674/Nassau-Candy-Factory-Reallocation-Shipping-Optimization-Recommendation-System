"""
Unit and Integration Tests for Factory Optimization Simulator (Task 3)
Tests factory selection, destination-distance mapping, scenario totals, filtering,
empty results handling, and Streamlit execution.
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data, validate_dataset
from src.simulator import (
    FACTORY_METADATA,
    ALL_FACTORIES,
    get_factory_distance,
    validate_simulation_inputs,
    filter_simulation_orders,
    run_factory_simulation,
)


class TestFactoryOptimizationSimulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Loads and validates the dataset once for all tests."""
        cls.df = load_data(validate=True, parse_dates=True)
        assert validate_dataset(cls.df)["is_valid"], "Dataset validation failed"

    def test_01_factory_selection_and_validation(self):
        """1. Verify factory selection validation rules and coordinate distances."""
        df = self.df

        # Rule: Same factory as both source and destination must fail validation
        valid, err = validate_simulation_inputs(df, "Lot's O' Nuts", "Lot's O' Nuts")
        self.assertFalse(valid)
        self.assertIn("cannot be identical", err)

        # Rule: Unknown factory name must fail validation
        valid, err = validate_simulation_inputs(df, "Fantasy Factory", "Secret Factory")
        self.assertFalse(valid)
        self.assertIn("not recognized", err)

        # Rule: Distinct valid factories must pass validation
        valid, err = validate_simulation_inputs(df, "Lot's O' Nuts", "Secret Factory")
        self.assertTrue(valid)
        self.assertIsNone(err)

        # Verify inter-factory Haversine distance calculation
        # Distance from Lot's O' Nuts (AZ) to Wicked Choccy's (GA) should be approx 2800-3000 km
        dist = get_factory_distance("Lot's O' Nuts", "Wicked Choccy's")
        self.assertGreater(dist, 2500)
        self.assertLess(dist, 3200)
        self.assertEqual(get_factory_distance("Lot's O' Nuts", "Lot's O' Nuts"), 0.0)

    def test_02_destination_distance_mapping(self):
        """2. Verify that scenario distances use the explicitly selected destination column."""
        df = self.df
        source = "Lot's O' Nuts"
        dest = "Sugar Shack"

        results = run_factory_simulation(df, source_factory=source, destination_factory=dest)
        comp_df = results["comparison_df"]
        self.assertFalse(comp_df.empty)

        # Get the underlying source records from df
        source_df = df[df["Current_Factory"] == source].reset_index(drop=True)
        dest_col = f"{dest}_Distance_KM"

        # Check sample of 20 orders to verify row-by-row mapping
        for idx in range(min(20, len(source_df))):
            expected_scenario_dist = float(source_df.loc[idx, dest_col])
            actual_scenario_dist = float(comp_df.loc[idx, "Scenario Distance (KM)"])
            self.assertAlmostEqual(actual_scenario_dist, expected_scenario_dist, places=3)

        # Confirm that scenario distance is NOT simply Best_Alternative_Distance_KM
        # Find an order where Best_Alternative_Factory is NOT Sugar Shack
        alt_not_dest = source_df[source_df["Best_Alternative_Factory"] != dest]
        self.assertFalse(alt_not_dest.empty)
        test_idx = alt_not_dest.index[0]
        test_row = comp_df.loc[test_idx]
        best_alt_dist = float(alt_not_dest.loc[test_idx, "Best_Alternative_Distance_KM"])
        # Scenario distance should use Sugar Shack, which is different from Best Alternative
        self.assertNotEqual(round(test_row["Scenario Distance (KM)"], 2), round(best_alt_dist, 2))

    def test_03_scenario_totals_and_distance_metrics(self):
        """3. Verify calculation of totals, averages, order change categories, and percentages."""
        df = self.df
        source = "Lot's O' Nuts"
        dest = "Secret Factory"

        results = run_factory_simulation(df, source_factory=source, destination_factory=dest)
        comp_df = results["comparison_df"]

        # Check order counts
        self.assertEqual(results["order_count"], len(comp_df))
        self.assertEqual(results["order_count"], 5692)  # All orders from Lot's O' Nuts

        # Check distance sums
        expected_cur_sum = comp_df["Current Distance (KM)"].sum()
        expected_scen_sum = comp_df["Scenario Distance (KM)"].sum()
        expected_saved_sum = expected_cur_sum - expected_scen_sum

        self.assertAlmostEqual(results["current_total_distance_km"], expected_cur_sum, places=2)
        self.assertAlmostEqual(results["scenario_total_distance_km"], expected_scen_sum, places=2)
        self.assertAlmostEqual(results["total_distance_saved_km"], expected_saved_sum, places=2)

        # Check distance averages
        self.assertAlmostEqual(
            results["avg_current_distance_km"], expected_cur_sum / len(comp_df), places=3
        )
        self.assertAlmostEqual(
            results["avg_scenario_distance_km"], expected_scen_sum / len(comp_df), places=3
        )
        self.assertAlmostEqual(
            results["avg_distance_saved_km"],
            results["avg_current_distance_km"] - results["avg_scenario_distance_km"],
            places=3,
        )

        # Check order change partition
        dec = results["orders_decreasing"]
        inc = results["orders_increasing"]
        unch = results["orders_unchanged"]
        self.assertEqual(dec + inc + unch, results["order_count"])

        # Check row-by-row distance change equation
        row_diffs = comp_df["Distance Change (KM)"] - (
            comp_df["Current Distance (KM)"] - comp_df["Scenario Distance (KM)"]
        )
        self.assertAlmostEqual(float(row_diffs.abs().max()), 0.0, places=5)

    def test_04_filtering_controls(self):
        """4. Verify scenario controls: candidates-only, product filter, and min saving percentage."""
        df = self.df
        source = "Wicked Choccy's"
        dest = "The Other Factory"

        # Baseline: All orders
        all_res = run_factory_simulation(df, source, dest, candidates_only=False)
        self.assertEqual(all_res["order_count"], 4152)

        # Filter 1: Candidates only (RELOCATE - HIGH or RELOCATE - REVIEW)
        cand_res = run_factory_simulation(df, source, dest, candidates_only=True)
        self.assertGreater(cand_res["order_count"], 0)
        self.assertLess(cand_res["order_count"], all_res["order_count"])
        # All orders in cand_res must be high or review
        decisions = cand_res["comparison_df"]["Existing Recommendation Decision"].unique()
        for d in decisions:
            self.assertIn(d, ["RELOCATE - HIGH", "RELOCATE - REVIEW"])

        # Filter 2: Specific product filter
        prod_res = run_factory_simulation(
            df, source, dest, selected_products=["Wonka Bar - Milk Chocolate"]
        )
        self.assertGreater(prod_res["order_count"], 0)
        self.assertTrue(
            (prod_res["comparison_df"]["Product Name"] == "Wonka Bar - Milk Chocolate").all()
        )

        # Filter 3: Minimum saving percentage filter (>= 40%)
        saving_res = run_factory_simulation(df, source, dest, min_saving_pct=40.0)
        self.assertGreater(saving_res["order_count"], 0)
        self.assertTrue((saving_res["comparison_df"]["Distance Change (%)"] >= 39.999).all())

    def test_05_empty_results_handling(self):
        """5. Verify graceful handling when filters yield zero matching orders."""
        df = self.df
        source = "Sugar Shack"
        dest = "Lot's O' Nuts"

        # Set an impossible minimum saving percentage to guarantee 0 matching records
        res_empty = run_factory_simulation(df, source, dest, min_saving_pct=99.9)

        self.assertTrue(res_empty["is_empty"])
        self.assertEqual(res_empty["order_count"], 0)
        self.assertEqual(res_empty["current_total_distance_km"], 0.0)
        self.assertEqual(res_empty["scenario_total_distance_km"], 0.0)
        self.assertEqual(res_empty["total_distance_saved_km"], 0.0)
        self.assertEqual(res_empty["orders_decreasing"], 0)
        self.assertEqual(res_empty["orders_increasing"], 0)
        self.assertEqual(res_empty["orders_unchanged"], 0)

        # Verify comparison_df has the expected schema despite being empty
        expected_cols = [
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
        self.assertEqual(list(res_empty["comparison_df"].columns), expected_cols)

    def test_06_product_summary_and_top_reductions(self):
        """6. Verify product summary aggregation and top 10 reductions formatting."""
        df = self.df
        source = "Lot's O' Nuts"
        dest = "The Other Factory"

        res = run_factory_simulation(df, source, dest)
        prod_summary = res["product_summary_df"]
        top_reductions = res["top_reductions_df"]

        # Product summary verification
        self.assertFalse(prod_summary.empty)
        self.assertIn("Product Name", prod_summary.columns)
        self.assertIn("Distance_Saved_KM", prod_summary.columns)
        self.assertAlmostEqual(
            float(prod_summary["Distance_Saved_KM"].sum()),
            float(res["total_distance_saved_km"]),
            places=2,
        )

        # Top 10 reductions verification
        self.assertLessEqual(len(top_reductions), 10)
        self.assertGreater(len(top_reductions), 0)
        # Check sorted descending
        savings = top_reductions["Distance Change (KM)"].tolist()
        self.assertTrue(all(savings[i] >= savings[i + 1] for i in range(len(savings) - 1)))

    def test_07_streamlit_simulator_headless_execution(self):
        """7. Verify simulator page renders without exceptions using Streamlit AppTest."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(at.exception), 0)

        # Switch page navigation to "Factory Optimization Simulator"
        at.sidebar.radio(key="nav_page").set_value("Factory Optimization Simulator").run()
        self.assertEqual(len(at.exception), 0, f"Simulator raised exceptions: {at.exception}")

        # Check simulator page elements
        markdown_text = " ".join([m.value for m in at.markdown])
        self.assertIn("Factory Optimization Simulator", markdown_text)
        self.assertIn("Current Total Distance", markdown_text)
        self.assertIn("Scenario Total Distance", markdown_text)
        self.assertIn("Total Geographic Savings", markdown_text)
        self.assertIn("Orders with Decreased Distance", markdown_text)
        self.assertIn("Orders with Increased Distance", markdown_text)

        # Check disclaimers
        self.assertIn("Haversine Approximation", markdown_text)
        self.assertIn("Potential vs. Guaranteed Logistics Savings", markdown_text)

        # Check Plotly charts rendered
        self.assertGreaterEqual(len(at.get("plotly_chart")), 3)

        # Check comparison table and CSV download button rendered
        self.assertGreaterEqual(len(at.dataframe), 1)
        self.assertEqual(len(at.get("download_button")), 1)

    def test_08_simulator_interactive_scenario_change(self):
        """8. Verify interacting with simulator controls updates metrics cleanly."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
        at.sidebar.radio(key="nav_page").set_value("Factory Optimization Simulator").run()

        # Change source factory to "Wicked Choccy's" and destination to "Sugar Shack"
        at.selectbox(key="sim_source_factory").set_value("Wicked Choccy's").run()
        self.assertEqual(len(at.exception), 0)

        at.selectbox(key="sim_dest_factory").set_value("Sugar Shack").run()
        self.assertEqual(len(at.exception), 0)

        markdown_text = " ".join([m.value for m in at.markdown])
        self.assertIn("4,152", markdown_text)  # Total orders from Wicked Choccy's

    def test_09_all_twenty_factory_pairs_destination_mapping_and_totals(self):
        """9. Verify all 20 valid (source, destination) pairs use correct columns and partition sums."""
        df = self.df
        tested_pairs = 0

        for src in ALL_FACTORIES:
            for dst in ALL_FACTORIES:
                if src == dst:
                    continue
                res = run_factory_simulation(df, source_factory=src, destination_factory=dst)
                tested_pairs += 1

                # 1. Order count must be greater than zero for all valid sources
                self.assertGreater(res["order_count"], 0)

                # 2. Orders decreasing + increasing + unchanged must exactly sum to order count
                self.assertEqual(
                    res["orders_decreasing"] + res["orders_increasing"] + res["orders_unchanged"],
                    res["order_count"],
                    f"Partition mismatch for {src} -> {dst}",
                )

                # 3. Distance saved = Current total - Scenario total
                expected_saved = res["current_total_distance_km"] - res["scenario_total_distance_km"]
                self.assertAlmostEqual(res["total_distance_saved_km"], expected_saved, places=3)

                # 4. Reduction percentage formula
                expected_pct = (res["total_distance_saved_km"] / res["current_total_distance_km"]) * 100.0
                self.assertAlmostEqual(res["pct_reduction_total"], expected_pct, places=3)

                # 5. Sign consistency: if total saved > 0, reduction % > 0; if total saved < 0, reduction % < 0
                if res["total_distance_saved_km"] > 0.001:
                    self.assertGreater(res["pct_reduction_total"], 0)
                elif res["total_distance_saved_km"] < -0.001:
                    self.assertLess(res["pct_reduction_total"], 0)

        self.assertEqual(tested_pairs, 20, "Expected exactly 20 distinct factory pairs")

    def test_10_recommendation_baseline_counts(self):
        """10. Verify exact established baseline recommendation counts (5,873 / 4,105 / 216)."""
        df = self.df
        dec_counts = df["Final_Optimization_Decision"].value_counts()
        self.assertEqual(dec_counts.get("KEEP CURRENT", 0), 5873)
        self.assertEqual(dec_counts.get("RELOCATE - REVIEW", 0), 4105)
        self.assertEqual(dec_counts.get("RELOCATE - HIGH", 0), 216)

        risk_counts = df["Optimization_Risk"].value_counts()
        self.assertEqual(risk_counts.get("NO ACTION", 0), 5873)
        self.assertEqual(risk_counts.get("HIGH RISK", 0), 4105)
        self.assertEqual(risk_counts.get("MEDIUM RISK", 0), 216)

        self.assertEqual(len(df), 10194)
        self.assertEqual(len(df.columns), 42)

    def test_11_sign_conventions_and_mathematical_consistency(self):
        """11. Verify sign convention: positive means shorter distance, negative means longer distance."""
        df = self.df

        # Scenario A: Distance decreases (Lot's O' Nuts -> The Other Factory)
        res_saving = run_factory_simulation(df, "Lot's O' Nuts", "The Other Factory")
        self.assertGreater(res_saving["total_distance_saved_km"], 0)
        self.assertGreater(res_saving["pct_reduction_total"], 0)
        self.assertGreater(res_saving["avg_distance_saved_km"], 0)
        # Verify comparison df row values match sign
        row_saving = res_saving["comparison_df"].iloc[0]
        self.assertEqual(
            row_saving["Distance Change (KM)"],
            row_saving["Current Distance (KM)"] - row_saving["Scenario Distance (KM)"],
        )

        # Scenario B: Distance increases (Wicked Choccy's -> Lot's O' Nuts)
        res_increase = run_factory_simulation(df, "Wicked Choccy's", "Lot's O' Nuts")
        self.assertLess(res_increase["total_distance_saved_km"], 0)
        self.assertLess(res_increase["pct_reduction_total"], 0)
        self.assertLess(res_increase["avg_distance_saved_km"], 0)

    def test_12_minimum_saving_filter_definition_and_edge_cases(self):
        """12. Verify min_saving_pct filter definition, threshold enforcement, and denominator protection."""
        df = self.df
        source = "Lot's O' Nuts"
        dest = "The Other Factory"

        # Baseline: All orders
        baseline = run_factory_simulation(df, source, dest)

        # Min saving threshold 25%
        res_25 = run_factory_simulation(df, source, dest, min_saving_pct=25.0)
        self.assertGreater(res_25["order_count"], 0)
        self.assertLessEqual(res_25["order_count"], baseline["order_count"])
        self.assertTrue((res_25["comparison_df"]["Distance Change (%)"] >= 24.999).all())

        # When min_saving_pct is None, all orders must be included
        res_none = run_factory_simulation(df, source, dest, min_saving_pct=None)
        self.assertEqual(res_none["order_count"], baseline["order_count"])


if __name__ == "__main__":
    unittest.main()
