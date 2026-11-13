# Nassau-Candy-Factory-Reallocation-Shipping-Optimization-Recommendation-System
Python-based factory reallocation and shipping optimization system featuring interactive dashboards, geographic analysis, and business impact insights using Streamlit, Pandas, and Plotly.
# 🍬 Nassau Candy – Factory Reallocation & Shipping Optimization Recommendation System

An interactive, Python-based decision-support application designed to analyze factory allocation, evaluate shipping distances, simulate factory reallocation scenarios, and identify potential logistics improvements.

## 📌 Project Overview

This project analyzes the Nassau Candy order dataset to identify opportunities for reducing customer-to-factory shipping distances through data-driven factory allocation recommendations.

The application provides interactive dashboards, geographic visualizations, factory simulation, and business impact analysis to support logistics and operational decision-making.

## 🎯 Project Objectives

- Analyze existing factory-to-customer allocation.
- Calculate distances between customers and factories.
- Identify potential factory reallocation opportunities.
- Compare current and alternative factory assignments.
- Visualize geographic distribution and optimization scenarios.
- Estimate illustrative business impact using configurable cost assumptions.

## 🚀 Application Modules

### 1. Executive Dashboard
- Overview of orders, sales, gross profit, and units.
- Factory allocation and recommendation analysis.
- Interactive charts and filters.
- Downloadable recommendation data.

### 2. Factory Optimization Simulator
- Select source and destination factories.
- Simulate alternative factory assignments.
- Compare current and proposed shipping distances.
- Analyze distance changes and export results.

### 3. Geographic Network
- Visualize factory locations and customer distribution.
- Explore geographic patterns across the United States and Canada.
- Compare current and proposed factory allocation scenarios.

### 4. Business Impact & Decision Intelligence
- Analyze potential distance reduction.
- Explore factory reallocation scenarios.
- Review product-level recommendations and priority scores.
- Estimate illustrative cost impact using an adjustable cost-per-kilometre assumption.
- Review compatibility warnings and decision recommendations.

## 📊 Key Project Results

| Metric | Result |
|---|---:|
| Total Orders Analyzed | 10,194 |
| Relocation Candidates | 4,321 |
| High-Priority Candidates | 216 |
| Candidate Current Distance | 12,203,265 km |
| Candidate Optimized Distance | 2,830,771 km |
| Potential Distance Reduction | 9,372,494 km |
| Aggregate Distance Reduction | 76.8% |

**Note:** These are modeled results for the selected relocation candidates, not guaranteed operational savings. Distance calculations use the Haversine formula (straight-line distance).

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Plotly
- **Web Application:** Streamlit
- **Geographic Analysis:** Haversine Distance Calculation, pgeocode
- **Testing:** Python unittest

## 📁 Project Structure

```text
factory-reallocation-optimization/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── processed/
│       └── cleaned_data.csv
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── simulator.py
│   ├── geo_network.py
│   └── business_impact.py
│
├── tests/
│   ├── test_audit_validation.py
│   ├── test_business_impact.py
│   ├── test_dashboard.py
│   ├── test_geo_network.py
│   └── test_simulator.py
│
└── screenshots/
    ├── executive_dashboard.png
    ├── factory_simulator.png
    ├── geographic_network.png
    └── business_impact.png
```

*Adjust the screenshot filenames or folder structure if your actual files are named differently.*

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/factory-reallocation-optimization.git
```

### 2. Navigate to the Project Folder

```bash
cd factory-reallocation-optimization
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

Open the application in your browser:

```text
http://localhost:8501
```

## 🧪 Run Tests

Execute the automated test suite:

```bash
python -m unittest discover -s tests -v
```

The project test suite contains **44 test cases**, covering dashboard functionality, simulator logic, geographic analysis, business impact, and validation.

## 📈 Optimization Methodology

1. Load and validate the processed order dataset.
2. Analyze customer and factory geographic coordinates.
3. Calculate distances using the Haversine formula.
4. Compare existing factory assignments with alternative factories.
5. Generate relocation recommendations using distance and business-related scoring factors.
6. Simulate alternative factory allocation scenarios.
7. Present geographic and illustrative business impact insights.

## ⚠️ Limitations

- Shipping distances are calculated using straight-line geographic distance, not actual road or transportation routes.
- Cost estimates use an assumed cost-per-kilometre value and are not actual freight savings.
- Factory capacity, machinery, labor availability, and production constraints require real-world verification.
- Product compatibility recommendations are heuristic and require operational validation.
- Lead-time prediction models were not sufficiently reliable for operational claims.

The application should therefore be treated as a **decision-support prototype**, not a production-validated logistics optimization system.

## 🔮 Future Improvements

- Integrate real road-network distances and routing APIs.
- Incorporate actual freight rates and transportation costs.
- Include factory capacity and production constraints.
- Improve optimization using mathematical programming techniques.
- Integrate real-time logistics and shipment information.
- Deploy the application to a cloud platform.

## 👨‍💻 Author

**Vedansh Kumar Sahu**

B.Tech – Computer Science and Engineering

GitHub: [Vedansh5674](https://github.com/Vedansh5674)

---

⭐ If you find this project useful, consider giving the repository a star!

