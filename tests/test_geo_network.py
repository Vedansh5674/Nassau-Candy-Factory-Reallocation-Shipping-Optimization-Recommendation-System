"""
Unit and Integration Tests for Geographic Network & Factory Location Analysis (Task 5)
Verifies factory coordinates, customer coordinate variation, mapping functions,
filters, simulation agreement, and state/city aggregation.
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data, validate_dataset
from src.simulator import (
    FACTORY_METADATA,
    ALL_FACTORIES,
    DISTANCE_TOLERANCE_KM,
    run_factory_simulation,
    get_factory_distance,
)
from src.geo_network import (
    get_factory_summary,
    validate_geo_coordinates,
    build_factory_location_map,
    aggregate_customer_locations,
    build_customer_distribution_map,
    build_assignment_comparison_map,
    build_state_concentration_chart,
    build_city_insights_table,
    FACTORY_COLORS,
)


class TestGeographicNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Loads and validates the dataset once for all tests."""
        cls.df = load_data(validate=True, parse_dates=True)

    def test_01_all_five_factory_coordinates_and_metadata(self):
        """Verify all five factory coordinates and metadata against project definitions."""
        expected_factories = {
            "Lot's O' Nuts": {"lat": 32.881893, "lon": -111.768036, "city": "Florence", "state": "AZ", "div": "Chocolate"},
            "Wicked Choccy's": {"lat": 32.076176, "lon": -81.088371, "city": "Savannah", "state": "GA", "div": "Chocolate"},
            "Sugar Shack": {"lat": 48.119140, "lon": -96.181150, "city": "Thief River Falls", "state": "MN", "div": "Sugar"},
            "Secret Factory": {"lat": 41.446333, "lon": -90.565487, "city": "Rock Island", "state": "IL", "div": "Sugar / Other"},
            "The Other Factory": {"lat": 35.117500, "lon": -89.971107, "city": "Memphis", "state": "TN", "div": "Sugar / Other"},
        }

        self.assertEqual(len(FACTORY_METADATA), 5)
        for name, exp in expected_factories.items():
            self.assertIn(name, FACTORY_METADATA)
            meta = FACTORY_METADATA[name]
            self.assertAlmostEqual(meta["latitude"], exp["lat"], places=5)
            self.assertAlmostEqual(meta["longitude"], exp["lon"], places=5)
            self.assertEqual(meta["city"], exp["city"])
            self.assertEqual(meta["state"], exp["state"])
            self.assertEqual(meta["division"], exp["div"])

    def test_02_customer_coordinates_variation_and_bounds(self):
        """Verify customer markers reflect genuine coordinate variation across US/Canada."""
        geo_health = validate_geo_coordinates(self.df)
        self.assertFalse(geo_health["has_single_centroid_error"])
        self.assertEqual(geo_health["lat_nulls"], 0)
        self.assertEqual(geo_health["lon_nulls"], 0)
        self.assertEqual(geo_health["unique_locations"], 654)

        # Latitude bounds in continental North America
        self.assertGreaterEqual(geo_health["lat_min"], 24.0)
        self.assertLessEqual(geo_health["lat_max"], 55.0)

        # Longitude bounds in North America
        self.assertGreaterEqual(geo_health["lon_min"], -130.0)
        self.assertLessEqual(geo_health["lon_max"], -50.0)

    def test_03_factory_summary_table_integrity(self):
        """Verify get_factory_summary correctly aggregates volume and financials."""
        summary = get_factory_summary(self.df)
        self.assertEqual(len(summary), 5)

        # Sum of assigned orders must equal 10,194
        self.assertEqual(summary["Assigned Orders"].sum(), 10194)
        # Sum of total units must equal 38,654
        self.assertEqual(summary["Total Units"].sum(), 38654)
        # Sum of sales must equal total sales in dataset
        self.assertAlmostEqual(summary["Total Sales ($)"].sum(), self.df["Sales"].sum(), places=1)
        # Sum of share % must be 100%
        self.assertAlmostEqual(summary["Network Share (%)"].sum(), 100.0, places=1)

    def test_04_customer_aggregation_and_filters(self):
        """Verify customer location aggregation respects factory, product, and decision filters."""
        # Unfiltered aggregation
        agg_all = aggregate_customer_locations(self.df)
        self.assertFalse(agg_all.empty)
        self.assertEqual(agg_all["order_count"].sum(), 10194)

        # Factory filter
        agg_wicked = aggregate_customer_locations(self.df, factory_filter="Wicked Choccy's")
        self.assertEqual(agg_wicked["order_count"].sum(), 4152)
        self.assertTrue((agg_wicked["Current_Factory"] == "Wicked Choccy's").all())

        # Decision filter (candidates only)
        agg_cand = aggregate_customer_locations(
            self.df, decision_filter="Reallocation Candidates Only (HIGH + REVIEW)"
        )
        self.assertEqual(agg_cand["order_count"].sum(), 4321)

        # Nonexistent filter combination returns empty dataframe cleanly
        agg_empty = aggregate_customer_locations(
            self.df, factory_filter="Sugar Shack", product_filter="Nonexistent Product XYZ"
        )
        self.assertTrue(agg_empty.empty)

    def test_05_assignment_comparison_agrees_with_simulator(self):
        """Verify geographic comparison metrics agree with the validated simulation engine."""
        source = "Lot's O' Nuts"
        dest = "Wicked Choccy's"

        sim = run_factory_simulation(
            df=self.df,
            source_factory=source,
            destination_factory=dest,
            candidates_only=False,
        )

        self.assertEqual(sim["order_count"], 5692)
        self.assertGreater(sim["current_total_distance_km"], 0)
        self.assertGreater(sim["scenario_total_distance_km"], 0)

        # Partition integrity
        total_partition = sim["orders_decreasing"] + sim["orders_increasing"] + sim["orders_unchanged"]
        self.assertEqual(total_partition, sim["order_count"])

        # Sign consistency: total distance saved = current - scenario
        expected_saved = sim["current_total_distance_km"] - sim["scenario_total_distance_km"]
        self.assertAlmostEqual(sim["total_distance_saved_km"], expected_saved, places=4)

    def test_06_build_maps_return_valid_plotly_figures(self):
        """Verify that map generation functions return valid Plotly Figures with correct traces."""
        # Factory Map
        fac_fig = build_factory_location_map(self.df)
        self.assertIsInstance(fac_fig, go.Figure)
        self.assertEqual(len(fac_fig.data), 5)  # 5 factory traces

        # Customer Distribution Map
        cust_fig = build_customer_distribution_map(self.df, factory_filter="Sugar Shack")
        self.assertIsInstance(cust_fig, go.Figure)
        self.assertGreaterEqual(len(cust_fig.data), 5)  # Customer traces + 5 factory markers

        # Assignment Comparison Map
        cmp_fig = build_assignment_comparison_map(
            df=self.df,
            source_factory="Secret Factory",
            destination_factory="The Other Factory",
            candidates_only=False,
            show_connections=True,
        )
        self.assertIsInstance(cmp_fig, go.Figure)
        self.assertGreater(len(cmp_fig.data), 0)

    def test_07_state_concentration_chart_and_city_table(self):
        """Verify state-level concentration chart and city insights table generation."""
        state_fig = build_state_concentration_chart(self.df, top_n=10)
        self.assertIsInstance(state_fig, go.Figure)
        self.assertGreater(len(state_fig.data), 0)

        city_df = build_city_insights_table(self.df, top_n=10)
        self.assertIsInstance(city_df, pd.DataFrame)
        self.assertEqual(len(city_df), 10)
        required_cols = [
            "City", "Country", "Orders", "Realloc Candidates", "Candidate %",
            "Total Sales ($)", "Potential Distance Saved (KM)", "Avg Saving %"
        ]
        for col in required_cols:
            self.assertIn(col, city_df.columns)

    def test_08_empty_filter_edge_cases(self):
        """Verify graceful handling when filters match zero records."""
        # Empty comparison map
        empty_fig = build_assignment_comparison_map(
            df=self.df,
            source_factory="Sugar Shack",
            destination_factory="Lot's O' Nuts",
            candidates_only=True,  # Sugar shack might have few or zero candidates
        )
        self.assertIsInstance(empty_fig, go.Figure)

        # Empty customer map
        empty_cust = build_customer_distribution_map(
            df=self.df,
            factory_filter="Sugar Shack",
            product_filter="Nonexistent 123",
        )
        self.assertIsInstance(empty_cust, go.Figure)


if __name__ == "__main__":
    unittest.main()
