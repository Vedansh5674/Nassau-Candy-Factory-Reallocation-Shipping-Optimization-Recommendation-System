"""
Unit and Integration Tests for Business Impact & Decision Intelligence (Task 7)
Tests freight cost proxy modeling, operational compatibility checking,
decision driver explanations, scenario execution, empty filter handling,
CSV exports, and Streamlit execution.
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
from src.business_impact import (
    calculate_freight_cost_proxy,
    check_operational_compatibility,
    generate_recommendation_drivers,
    run_global_business_impact,
)
from src.simulator import run_factory_simulation, DISTANCE_TOLERANCE_KM


class TestBusinessImpactDecisionIntelligence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Loads and validates the baseline dataset once for all tests."""
        cls.df = load_data(validate=True, parse_dates=True)

    # ------------------------------------------------------------------
    # 1. Freight Cost Proxy Calculations
    # ------------------------------------------------------------------
    def test_01_freight_cost_proxy_calculations(self):
        """Verify illustrative freight-cost proxy math, differences, and edge cases."""
        # Standard positive saving scenario
        cur_km = 1000.0
        scen_km = 600.0
        rate = 1.50
        res = calculate_freight_cost_proxy(cur_km, scen_km, rate)

        self.assertEqual(res["assumed_cost_per_km"], 1.50)
        self.assertAlmostEqual(res["baseline_cost_proxy"], 1500.0)
        self.assertAlmostEqual(res["scenario_cost_proxy"], 900.0)
        self.assertAlmostEqual(res["estimated_difference"], 600.0)
        self.assertAlmostEqual(res["pct_difference"], 40.0)

        # Distance increase scenario (negative difference)
        res_inc = calculate_freight_cost_proxy(500.0, 800.0, 2.00)
        self.assertAlmostEqual(res_inc["baseline_cost_proxy"], 1000.0)
        self.assertAlmostEqual(res_inc["scenario_cost_proxy"], 1600.0)
        self.assertAlmostEqual(res_inc["estimated_difference"], -600.0)
        self.assertAlmostEqual(res_inc["pct_difference"], -60.0)

        # Zero rate scenario
        res_zero = calculate_freight_cost_proxy(1000.0, 500.0, 0.0)
        self.assertEqual(res_zero["baseline_cost_proxy"], 0.0)
        self.assertEqual(res_zero["scenario_cost_proxy"], 0.0)
        self.assertEqual(res_zero["estimated_difference"], 0.0)
        self.assertEqual(res_zero["pct_difference"], 0.0)

        # Negative rate must raise ValueError
        with self.assertRaises(ValueError):
            calculate_freight_cost_proxy(1000.0, 500.0, -1.0)

    # ------------------------------------------------------------------
    # 2. Operational Compatibility Checks
    # ------------------------------------------------------------------
    def test_02_operational_compatibility_checks(self):
        """Verify facility specialization checking against product divisions."""
        # Chocolate product to chocolate plant (Lot's O' Nuts)
        res1 = check_operational_compatibility("Chocolate", "Lot's O' Nuts")
        self.assertEqual(res1["status"], "COMPATIBLE")
        self.assertTrue(res1["is_compatible"])

        # Chocolate product to chocolate plant (Wicked Choccy's)
        res2 = check_operational_compatibility("Chocolate", "Wicked Choccy's")
        self.assertEqual(res2["status"], "COMPATIBLE")

        # Chocolate product to Sugar-only plant (Sugar Shack) -> Incompatible
        res3 = check_operational_compatibility("Chocolate", "Sugar Shack")
        self.assertEqual(res3["status"], "INCOMPATIBLE")
        self.assertFalse(res3["is_compatible"])
        self.assertIn("Cross-Division Mismatch", res3["message"])

        # Sugar product to Sugar Shack -> Compatible
        res4 = check_operational_compatibility("Sugar", "Sugar Shack")
        self.assertEqual(res4["status"], "COMPATIBLE")

        # Other product to Secret Factory (Sugar / Other) -> Compatible
        res5 = check_operational_compatibility("Other", "Secret Factory")
        self.assertEqual(res5["status"], "COMPATIBLE")

        # Unknown factory
        res_unk = check_operational_compatibility("Chocolate", "Nonexistent Facility")
        self.assertEqual(res_unk["status"], "UNKNOWN")

    # ------------------------------------------------------------------
    # 3. Decision Driver Explanations
    # ------------------------------------------------------------------
    def test_03_recommendation_driver_generation(self):
        """Verify decision driver explanation text generation from underlying factors."""
        row_high = pd.Series({
            "Distance_Score": 85.0,
            "Alternative_Distance_Saving_KM": 1200.0,
            "Alternative_Distance_Saving_Pct": 75.0,
            "Burden_Reduction_Score": 25.0,
            "Profit_Margin_Pct": 65.0,
            "Units": 12,
        })
        explanation = generate_recommendation_drivers(row_high)
        self.assertIn("distance reduction", explanation.lower())
        self.assertIn("burden", explanation.lower())
        self.assertIn("margin", explanation.lower())
        self.assertIn("volume", explanation.lower())

        row_low = pd.Series({
            "Distance_Score": 10.0,
            "Alternative_Distance_Saving_KM": 50.0,
            "Alternative_Distance_Saving_Pct": 5.0,
            "Burden_Reduction_Score": 2.0,
            "Profit_Margin_Pct": 20.0,
            "Units": 1,
        })
        exp_low = generate_recommendation_drivers(row_low)
        self.assertTrue(len(exp_low) > 0)

    # ------------------------------------------------------------------
    # 4. Global Business Impact Scenario
    # ------------------------------------------------------------------
    def test_04_global_business_impact_candidates_only(self):
        """Verify global business impact evaluation on reallocation candidates."""
        df = self.df
        res = run_global_business_impact(df, candidates_only=True, assumed_cost_per_km=1.50)

        self.assertFalse(res["is_empty"])
        self.assertEqual(res["order_count"], 4321)
        self.assertEqual(res["realloc_candidates"], 4321)
        self.assertEqual(res["high_priority_candidates"], 216)

        # Distance checks
        self.assertAlmostEqual(res["current_total_distance_km"], 12203265.14, places=1)
        self.assertAlmostEqual(res["total_distance_saved_km"], 9372494.19, places=1)
        self.assertAlmostEqual(res["aggregate_reduction_pct"], 76.80, places=1)
        self.assertAlmostEqual(res["mean_order_saving_pct"], 77.75, delta=0.5)

        # Partition sum
        dec = res["orders_decreasing"]
        inc = res["orders_increasing"]
        unch = res["orders_unchanged"]
        self.assertEqual(dec + inc + unch, 4321)
        self.assertEqual(dec, 4321)  # All candidates reduce distance

        # Freight proxy
        proxy = res["freight_proxy"]
        expected_diff = 9372494.19 * 1.50
        self.assertAlmostEqual(proxy["estimated_difference"], expected_diff, places=0)

        # Prioritized table
        p_df = res["prioritized_df"]
        self.assertEqual(len(p_df), 4321)
        self.assertIn("Operational_Compatibility", p_df.columns)
        self.assertIn("Key_Decision_Drivers", p_df.columns)
        self.assertIn("Illustrative_Cost_Difference_$", p_df.columns)

    def test_05_global_business_impact_all_orders(self):
        """Verify global business impact evaluation across all 10,194 orders."""
        df = self.df
        res = run_global_business_impact(df, candidates_only=False, assumed_cost_per_km=2.00)

        self.assertEqual(res["order_count"], 10194)
        self.assertAlmostEqual(res["current_total_distance_km"], 20480868.06, places=1)
        self.assertAlmostEqual(res["total_distance_saved_km"], 9446481.72, places=1)
        self.assertAlmostEqual(res["aggregate_reduction_pct"], 46.12, places=1)
        self.assertAlmostEqual(res["mean_order_saving_pct"], -18.06, delta=0.5)

        # Partition sum
        dec = res["orders_decreasing"]
        inc = res["orders_increasing"]
        unch = res["orders_unchanged"]
        self.assertEqual(dec + inc + unch, 10194)

    # ------------------------------------------------------------------
    # 5. Filtering and Empty Dataset Handling
    # ------------------------------------------------------------------
    def test_06_filtering_and_empty_dataset_handling(self):
        """Verify filtering by products and safe handling of empty filter selections."""
        df = self.df
        single_prod = "Wonka Bar - Milk Chocolate"

        res_filtered = run_global_business_impact(
            df, candidates_only=True, selected_products=[single_prod]
        )
        self.assertGreater(res_filtered["order_count"], 0)
        self.assertTrue((res_filtered["prioritized_df"]["Product Name"] == single_prod).all())

        # Empty selection
        res_empty = run_global_business_impact(
            df, candidates_only=True, selected_products=["Nonexistent Candy XYZ"]
        )
        self.assertTrue(res_empty["is_empty"])
        self.assertEqual(res_empty["order_count"], 0)
        self.assertEqual(res_empty["total_distance_saved_km"], 0.0)
        self.assertEqual(res_empty["orders_decreasing"], 0)
        self.assertEqual(res_empty["orders_increasing"], 0)
        self.assertEqual(res_empty["orders_unchanged"], 0)

    # ------------------------------------------------------------------
    # 6. Streamlit Headless Execution via AppTest
    # ------------------------------------------------------------------
    def test_07_streamlit_business_impact_headless_execution(self):
        """Verify Business Impact page renders cleanly in Streamlit without exceptions."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(at.exception), 0)

        # Navigate to Business Impact & Decision Intelligence
        at.sidebar.radio(key="nav_page").set_value("Business Impact & Decision Intelligence").run()
        self.assertEqual(len(at.exception), 0, f"AppTest raised exceptions: {at.exception}")

        markdown_text = " ".join([m.value for m in at.markdown])
        self.assertIn("Business Impact & Decision Intelligence", markdown_text)
        self.assertIn("Evaluated Orders", markdown_text)
        self.assertIn("Current Total Distance", markdown_text)
        self.assertIn("Total Geographic Savings", markdown_text)
        self.assertIn("Assumed Rate Benchmark", markdown_text)
        self.assertIn("Baseline Cost Proxy", markdown_text)
        self.assertIn("Illustrative Difference", markdown_text)
        self.assertIn("Key Operational Constraints", markdown_text)
        self.assertIn("Recommended Next Actions", markdown_text)

        # Check disclaimers
        self.assertIn("Illustrative Freight Proxy Only", markdown_text)
        self.assertIn("Heuristic Prioritization", markdown_text)

        # Check Plotly charts and dataframes
        self.assertGreaterEqual(len(at.get("plotly_chart")), 1)
        self.assertGreaterEqual(len(at.dataframe), 1)
        self.assertEqual(len(at.get("download_button")), 1)

    def test_08_streamlit_interactive_cost_parameter_change(self):
        """Verify interacting with assumed cost parameter updates metrics without errors."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
        at.sidebar.radio(key="nav_page").set_value("Business Impact & Decision Intelligence").run()

        # Update cost per km to 3.00
        at.number_input(key="bi_cost_per_km").set_value(3.00).run()
        self.assertEqual(len(at.exception), 0)

        markdown_text = " ".join([m.value for m in at.markdown])
        self.assertIn("$3.00 / km", markdown_text)


if __name__ == "__main__":
    unittest.main()
