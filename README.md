# Nassau Candy Factory Reallocation & Shipping Optimization Recommendation System

## Overview

An interactive Streamlit decision-support prototype for reviewing Nassau Candy order-to-factory assignments. It combines order, product, financial, geographic, distance, and precomputed recommendation fields to explore whether assigning an order to another factory could reduce its straight-line distance to customers. Recommendations are analytical prompts for human review, not execution instructions.

## Business problem and objectives

A multi-factory distributor needs a consistent way to examine whether current factory assignments may create avoidable geographic shipping burden. This project aims to:

- Summarize order assignments, products, sales, units, and recommendation categories.
- Compare current and alternative factory distances at order and scenario level.
- Help users explore potential reallocations and their distance implications.
- Surface geographic patterns, recommendation drivers, and operational review needs.

## Dataset and preprocessing

The included processed dataset is `data/processed/cleaned_data.csv`. The loader expects 42 columns covering order identifiers and dates, customer geography, product and financial measures, five factory-distance fields, recommendation scores, and decision/risk labels. `src/data_loader.py` validates the required schema and categorical labels, reports nulls and score-range warnings, and parses order and ship dates when loading. It resolves the processed path first and falls back to root-level `cleaned_data.csv`.

The repository also includes `data/processed/cleaned_data_backup.csv` and `scripts/repair_dataset.py`. The repair script documents a coordinate and distance recalculation workflow using postal geography and vectorized Haversine calculations; it is a separate utility and is not run automatically by the application. Keep the processed CSV with the project when running the app.

## Technologies

- Python
- Streamlit for the interactive app
- Pandas and NumPy for data handling and calculations
- Plotly Express and Graph Objects for interactive charts and maps
- Scikit-learn and pgeocode are listed in `requirements.txt`; pgeocode is used by the dataset repair utility.

## Application pages

1. **Executive Dashboard** — filters records and presents decision KPIs, charts, and recommendation summaries.
2. **Factory Optimization Simulator** — selects a source and destination factory, optionally filters to recommended candidates/products/minimum distance change, and compares scenario distance totals and order-level changes.
3. **Geographic Network** — visualizes the five factory locations, customer geography, factory assignments, and network patterns.
4. **Business Impact & Decision Intelligence** — examines recommendation priorities and drivers, operational compatibility indicators, reviewer actions, and user-parameterized freight cost scenarios.

## Distance and recommendation logic

Factory coordinates are defined in `src/simulator.py`. The helper `get_factory_distance` computes direct great-circle distance with the Haversine formula using an Earth radius of 6,371 km. The simulator compares the selected destination's distance column with the current/source distance for the filtered orders; it does not calculate road routes, actual carrier mileage, or travel times. A 0.001 km tolerance is used to classify unchanged distances.

The cleaned dataset contains the recommendation scores and labels consumed by the dashboard. The application exposes distance, burden-reduction, profitability, volume, and aggregate recommendation score fields. The repair utility recalculates nearest alternatives and applies its recommendation scoring/decision process. In the live app, treat those precomputed labels as screening categories: `RELOCATE - HIGH` and `RELOCATE - REVIEW` identify candidates for further review, while `KEEP CURRENT` indicates no proposed move under that analysis. The simulator allows scenario-specific comparisons and does not itself establish that a move is operationally feasible.

## Dataset findings

The included processed CSV contains **10,194 order records**. Its decision labels count:

- **5,873** `KEEP CURRENT`
- **4,105** `RELOCATE - REVIEW`
- **216** `RELOCATE - HIGH`
- **4,321** combined relocation candidates (`RELOCATE - REVIEW` plus `RELOCATE - HIGH`)

These are counts in the bundled dataset, not measured realized savings. Freight cost outputs are illustrative scenario estimates derived from straight-line distances and a user-supplied cost-per-kilometer assumption; they are not verified freight savings, carrier quotes, invoices, or guaranteed outcomes.

## Windows setup

1. Install Python 3.10 or newer and open PowerShell.
2. Change to the project directory:

   ```powershell
   cd "C:\path\to\factory-reallocation-optimization"
   ```

3. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

If PowerShell blocks activation, use the environment's interpreter directly (for example, `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`).

## Run the application

From the project root with the environment active:

```powershell
python -m streamlit run app.py
```

Streamlit prints a local URL in the terminal; open it in a browser. Stop the server with `Ctrl+C`.

## Run automated tests

From the project root with dependencies installed:

```powershell
python -m unittest discover -s tests -v
```

This invokes the test modules under `tests/`. Test results depend on the local environment and have not been asserted by this README.

## Project structure

```text
factory-reallocation-optimization/
├── app.py
├── requirements.txt
├── cleaned_data.csv                 # root fallback dataset
├── data/
│   └── processed/
│       ├── cleaned_data.csv         # primary processed dataset
│       └── cleaned_data_backup.csv
├── scripts/
│   └── repair_dataset.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── simulator.py
│   ├── geo_network.py
│   └── business_impact.py
└── tests/
    ├── test_audit_validation.py
    ├── test_business_impact.py
    ├── test_dashboard.py
    ├── test_geo_network.py
    └── test_simulator.py
```

## Limitations and future improvements

- Straight-line geographic distances are a proxy and can differ substantially from road routes, travel time, and actual shipment paths.
- Freight cost outputs are illustrative scenarios; actual freight rates, shipment weights, carrier terms, accessorial charges, and realized savings are not established here.
- Factory capacity, equipment and product compatibility, production costs, labor, inventory, lead times, service levels, and other operational constraints require validation with business data and operations teams before any reassignment.
- Recommendation labels are based on the supplied processed data and its scoring workflow; they do not prove a move is feasible or beneficial in practice.
- Future work could integrate road-network routing and carrier rates, validate capacity and production constraints, quantify uncertainty, add scenario audit/export controls, and refresh recommendations from governed source data.
