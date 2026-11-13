"""
Unit and Integration Tests for Executive Dashboard (Task 2)
Tests data loading, KPI calculations, Plotly charts, filters, and Streamlit execution.
"""

import sys
import unittest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data, validate_dataset, get_dataset_summary


class TestExecutiveDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Loads and validates the dataset once for all tests."""
        cls.df = load_data(validate=True, parse_dates=True)
        cls.validation_report = validate_dataset(cls.df)

    def test_01_dataset_validation(self):
        """1. Verify dataset schema, record count, and column count."""
        self.assertTrue(self.validation_report["is_valid"], "Dataset validation failed")
        self.assertEqual(len(self.df), 10194, "Expected exactly 10,194 rows")
        self.assertEqual(len(self.df.columns), 42, "Expected exactly 42 columns")
        self.assertEqual(len(self.validation_report["missing_columns"]), 0)

    def test_02_kpi_calculations_unfiltered(self):
        """2. Verify all 8 KPIs calculated dynamically on the unfiltered dataset."""
        df = self.df
        total_orders = len(df)
        total_sales = float(df["Sales"].sum())
        total_profit = float(df["Gross Profit"].sum())
        total_units = int(df["Units"].sum())

        realloc_candidates = int(
            df["Final_Optimization_Decision"].isin(["RELOCATE - HIGH", "RELOCATE - REVIEW"]).sum()
        )
        high_priority = int((df["Final_Optimization_Decision"] == "RELOCATE - HIGH").sum())
        keep_current = int((df["Final_Optimization_Decision"] == "KEEP CURRENT").sum())

        total_distance_saving_km = float(df["Alternative_Distance_Saving_KM"].sum())
        avg_distance_saving_pct = float(df["Alternative_Distance_Saving_Pct"].mean())

        # Assertions
        self.assertEqual(total_orders, 10194)
        self.assertAlmostEqual(total_sales, 141783.63, places=2)
        self.assertAlmostEqual(total_profit, 93442.80, places=2)
        self.assertEqual(total_units, 38654)
        self.assertEqual(realloc_candidates, 4321)
        self.assertEqual(high_priority, 216)
        self.assertEqual(keep_current, 5873)
        self.assertEqual(realloc_candidates + keep_current, total_orders)
        self.assertGreater(total_distance_saving_km, 9000000)
        self.assertAlmostEqual(avg_distance_saving_pct, -18.06, delta=0.5)

    def test_03_filtered_kpi_updates(self):
        """3. Verify that KPIs dynamically recalculate on filtered subsets."""
        # Scenario: Filter by "KEEP CURRENT" only
        subset_keep = self.df[self.df["Final_Optimization_Decision"] == "KEEP CURRENT"]
        self.assertEqual(len(subset_keep), 5873)
        realloc_candidates = subset_keep["Final_Optimization_Decision"].isin(
            ["RELOCATE - HIGH", "RELOCATE - REVIEW"]
        ).sum()
        high_priority = (subset_keep["Final_Optimization_Decision"] == "RELOCATE - HIGH").sum()
        self.assertEqual(realloc_candidates, 0)
        self.assertEqual(high_priority, 0)

        # Scenario: Filter by single factory "Lot's O' Nuts"
        subset_factory = self.df[self.df["Current_Factory"] == "Lot's O' Nuts"]
        self.assertEqual(len(subset_factory), 5692)
        self.assertAlmostEqual(float(subset_factory["Sales"].sum()), 76340.15, places=2)

    def test_04_summary_table_columns_and_sorting(self):
        """4. Verify that summary table columns and sorting match requirements."""
        required_cols = [
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
        for col in required_cols:
            self.assertIn(col, self.df.columns, f"Missing required table column: {col}")

        # Check sorting by Recommendation_Score descending
        table_df = self.df[required_cols].sort_values(by="Recommendation_Score", ascending=False)
        scores = table_df["Recommendation_Score"].tolist()
        self.assertTrue(all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)))

    def test_05_plotly_charts_generation(self):
        """5. Verify that all 5 Plotly charts generate without error."""
        import plotly.express as px
        df = self.df

        # Chart 1: Distribution of final optimization decisions
        dec_counts = df["Final_Optimization_Decision"].value_counts().reset_index()
        dec_counts.columns = ["Decision", "Count"]
        fig1 = px.pie(dec_counts, names="Decision", values="Count", hole=0.45)
        self.assertEqual(len(fig1.data[0].labels), 3)

        # Chart 2: Distribution of optimization risk categories
        risk_counts = df["Optimization_Risk"].value_counts().reset_index()
        risk_counts.columns = ["Risk_Level", "Count"]
        fig2 = px.bar(risk_counts, x="Risk_Level", y="Count")
        self.assertEqual(len(fig2.data[0].x), 3)

        # Chart 3: Average current vs alternative distance by current factory
        dist_factory = (
            df.groupby("Current_Factory")[
                ["Current_Factory_Distance_KM", "Best_Alternative_Distance_KM"]
            ]
            .mean()
            .reset_index()
        )
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name="Current", x=dist_factory["Current_Factory"], y=dist_factory["Current_Factory_Distance_KM"]))
        fig3.add_trace(go.Bar(name="Alt", x=dist_factory["Current_Factory"], y=dist_factory["Best_Alternative_Distance_KM"]))
        self.assertEqual(len(fig3.data), 2)
        self.assertEqual(len(dist_factory), 5)

        # Chart 4: Top 10 products by potential distance savings
        top_products = (
            df.groupby("Product Name")["Alternative_Distance_Saving_KM"]
            .sum()
            .reset_index()
            .sort_values(by="Alternative_Distance_Saving_KM", ascending=True)
            .tail(10)
        )
        fig4 = px.bar(top_products, x="Alternative_Distance_Saving_KM", y="Product Name", orientation="h")
        self.assertEqual(len(fig4.data[0].y), 10)

        # Chart 5: Current factory vs recommended alternative factory
        flow_df = df.groupby(["Current_Factory", "Best_Alternative_Factory"]).size().reset_index(name="Order_Count")
        fig5 = px.bar(flow_df, x="Current_Factory", y="Order_Count", color="Best_Alternative_Factory")
        self.assertGreater(len(fig5.data), 0)

    def test_06_streamlit_app_headless_execution(self):
        """6. Run the Streamlit application using AppTest to confirm error-free execution."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(at.exception), 0, f"AppTest raised exceptions: {at.exception}")

        # Check all 5 Plotly charts rendered
        self.assertEqual(len(at.get("plotly_chart")), 5, f"Expected 5 plotly charts, found {len(at.get('plotly_chart'))}")

        # Check summary dataframe exists
        self.assertGreaterEqual(len(at.dataframe), 1, "Expected summary dataframe to be rendered")

        # Check download button exists
        self.assertEqual(len(at.get("download_button")), 1, "Expected CSV download button")

        # Check KPI labels and values rendered in markdown
        markdown_text = " ".join([m.value for m in at.markdown])
        expected_kpis = [
            "Total Orders",
            "Total Sales",
            "Total Gross Profit",
            "Total Units",
            "Reallocation Candidates",
            "High-Priority Recommendations",
            "Potential Geographic Distance Savings",
            "Average Potential Distance Saving Percentage",
        ]
        for kpi in expected_kpis:
            self.assertIn(kpi, markdown_text, f"KPI title '{kpi}' not found in rendered dashboard markdown")

        # Check business limitations & disclaimer content
        self.assertIn("Haversine", markdown_text)
        self.assertIn("Potential vs. Guaranteed Savings", markdown_text)
        self.assertIn("Decision Support Framework", markdown_text)

    def test_07_interactive_filter_changes_in_app(self):
        """7. Verify changing filters in Streamlit AppTest updates output without error."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()

        # Change factory filter to only ["Secret Factory"]
        at.sidebar.multiselect(key="filter_factory").set_value(["Secret Factory"]).run()
        self.assertEqual(len(at.exception), 0)

        # Check that total orders in markdown reflects Secret Factory count (217)
        markdown_text = " ".join([m.value for m in at.markdown])
        self.assertIn("217", markdown_text)

    def test_08_geographic_coordinates_and_distance_variation(self):
        """8. Verify customer coordinates are not collapsed and distance columns contain meaningful variation."""
        df = self.df
        self.assertGreater(df["Latitude"].nunique(), 600, "Expected > 600 unique latitudes")
        self.assertGreater(df["Longitude"].nunique(), 600, "Expected > 600 unique longitudes")
        self.assertEqual(df["Latitude"].isna().sum(), 0, "Expected 0 missing latitudes")
        self.assertEqual(df["Longitude"].isna().sum(), 0, "Expected 0 missing longitudes")

        # Factory distances must have realistic spread across geographical locations
        factory_names = [
            "Lot's O' Nuts",
            "Wicked Choccy's",
            "Sugar Shack",
            "Secret Factory",
            "The Other Factory",
        ]
        for factory in factory_names:
            col = f"{factory}_Distance_KM"
            self.assertIn(col, df.columns)
            min_val = df[col].min()
            max_val = df[col].max()
            self.assertLess(min_val, 200, f"{factory} minimum distance should be < 200 km")
            self.assertGreater(max_val, 3000, f"{factory} maximum distance should be > 3000 km")
            self.assertGreater(df[col].nunique(), 500, f"{factory} distances should vary across locations")

        # Best alternative factory must be distributed realistically across all 5 factories
        self.assertEqual(
            df["Best_Alternative_Factory"].nunique(),
            5,
            "Expected all 5 factories to be represented as alternatives",
        )
        for count in df["Best_Alternative_Factory"].value_counts():
            self.assertGreater(
                count, 1000, "Expected each factory to have > 1,000 recommendations as alternative"
            )


if __name__ == "__main__":
    unittest.main()
