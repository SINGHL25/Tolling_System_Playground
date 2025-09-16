# Tolling_System_Playground
<img width="1024" height="1024" alt="Generated Image September 17, 2025 - 5_44AM" src="https://github.com/user-attachments/assets/ae60106e-7341-4bde-9821-3cb437f66b78" />

```plain text
Tolling_System_Playground/
│
├── README.md                       # Overview, goals, quickstart
├── requirements.txt                # Python deps (pandas, scikit-learn, streamlit, matplotlib, seaborn, plotly)
├── LICENSE                         # Open source license
│
├── docs/                           # Knowledge base & theory
│   ├── 01_overview.md              # Intro to tolling systems
│   ├── 02_passage_data.md          # Vehicle passages (fields, logs, examples)
│   ├── 03_transaction_data.md      # Transactions, revenue mapping
│   ├── 04_ivdc_data.md             # IVDC flows, errors, validations
│   ├── 05_statistics_kpis.md       # Revenue, congestion, traffic flow KPIs
│   ├── 06_air_pollution.md         # Emissions, pollution monitoring
│   ├── 07_health_status.md         # Roadside systems health, uptime, alarms
│   ├── 08_ml_models.md             # Predictive ML (traffic, revenue, anomalies)
│   ├── 09_future_scope.md          # Smart city, EV charging, AI tolling
│   └── glossary.md
│
├── data/                           # Datasets
│   ├── raw/
│   │   ├── passages.xlsx           # Vehicle passage records
│   │   ├── transactions.csv        # Revenue, toll booth transactions
│   │   ├── ivdc_passages.csv       # IVDC logs
│   │   └── roadside_sensors.csv    # Environmental/health sensors
│   ├── processed/
│   │   ├── traffic_summary.parquet
│   │   ├── revenue_per_lane.csv
│   │   ├── ivdc_cleaned.csv
│   │   └── pollution_stats.csv
│   └── external/                   # Govt traffic DBs, open datasets
│
├── src/                            # Core Python library
│   ├── __init__.py
│   ├── passage_analyzer.py         # Parse passage/IVDC logs
│   ├── transaction_analyzer.py     # Revenue and tolling data
│   ├── congestion_estimator.py     # KPIs: congestion, avg wait time
│   ├── pollution_analyzer.py       # Air quality & emissions impact
│   ├── health_monitor.py           # Roadside system status, uptime
│   ├── stats_visualizer.py         # KPI graphs, trends
│   ├── ml_models.py                # ML prediction (traffic, anomalies, revenue)
│   └── utils.py                    # Shared helpers
│
├── notebooks/                      # Jupyter notebooks for exploration
│   ├── 01_Passage_Data_Analysis.ipynb
│   ├── 02_Transaction_Analytics.ipynb
│   ├── 03_IVDC_Validation.ipynb
│   ├── 04_Revenue_KPIs.ipynb
│   ├── 05_Congestion_Prediction.ipynb
│   ├── 06_Pollution_Impact.ipynb
│   ├── 07_System_Health.ipynb
│   └── 08_Traffic_Revenue_Prediction.ipynb
│
├── models/                         # ML models
│   ├── trained/
│   │   ├── traffic_forecast.pkl
│   │   ├── revenue_predictor.pkl
│   │   └── anomaly_detector.pkl
│   └── experiments/
│       ├── logs.csv
│       └── tuning_results.json
│
├── streamlit_app/                  # Dashboards for toll operators
│   ├── app.py                      # Entry point
│   ├── pages/
│   │   ├── 1_Passage_Explorer.py   # Explore passages, IVDC logs
│   │   ├── 2_Transaction_Dashboard.py # Revenue analytics
│   │   ├── 3_Congestion_Monitor.py # Congestion KPIs, traffic heatmap
│   │   ├── 4_Pollution_Impact.py   # Air pollution & emissions
│   │   ├── 5_System_Health.py      # Device/system uptime, alerts
│   │   ├── 6_Traffic_Prediction.py # ML-based forecasting
│   │   └── 7_Future_Scope.py       # Smart tolling ideas
│   └── utils/
│       └── visual_helpers.py
│
├── api/                            # REST API (FastAPI/Flask)
│   ├── main.py
│   ├── routes/
│   │   ├── passages.py
│   │   ├── transactions.py
│   │   ├── congestion.py
│   │   ├── pollution.py
│   │   └── health.py
│   └── schemas/
│       └── passage_schema.py
│
├── cli.py                          # CLI tool: quick stats & reports
│
├── examples/                       # Example reports & configs
│   ├── sample_passage.json
│   ├── revenue_report.xlsx
│   └── congestion_summary.md
│
├── tests/                          # Unit tests
│   ├── test_passage_analyzer.py
│   ├── test_transaction_analyzer.py
│   ├── test_congestion_estimator.py
│   ├── test_pollution_analyzer.py
│   ├── test_health_monitor.py
│   └── test_ml_models.py
│
└── images/                         # Infographics & dashboards
    ├── toll_flow.png
    ├── revenue_trends.png
    ├── congestion_map.png
    └── pollution_chart.png


```
