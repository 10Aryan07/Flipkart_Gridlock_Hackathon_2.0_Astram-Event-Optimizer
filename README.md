# ASTraM Event Optimizer
**Flipkart Gridlock Hackathon 2.0 - Phase 2 Submission**

The ASTraM Event Optimizer is a real-time, machine-learning-driven command center designed to shift municipal traffic management from *reactive* to *proactive*. By predicting the spatiotemporal impact of localized events(e.g., waterlogging, public events) before they escalate, ASTraM instantly generates a mathematical Dispatch Manifest to optimize police manpower and physical resource deployment.

## System Architecture & ML Strategy

To ensure maximum scalability and performance, this project utilizes a fully decoupled architecture:

* **Frontend(Streamlit + PyDeck):** Handles real-time 3D geospatial rendering, user inputs and metric visualization.
* **Backend API(FastAPI + Uvicorn):** A lightning-fast, asynchronous bridge that houses the live intelligence engine.
* **Machine Learning Pipeline(Scikit-Learn):** * **Phase 1 (Archive):** Our initial data science heavy-lifting involved training complex offline ensembles(LightGBM/CatBoost) to establish baseline predictions and feature importance. These batch-processing notebooks are preserved in the `archive_phase1/` folder.
 * **Phase 2 (Live Inference):** For real-time API deployment, we transitioned to a lightweight, structural **Random Forest Regressor**. This live model dynamically calculates severity scores on the fly rather than relying on static historical lookups.

## Core Intelligence Features

### 1. Cyclical Feature Engineering
The engine does not treat time as a static integer. Datetime inputs are mathematically transformed into continuous trigonometric sine and cosine waves:
`hour_sin = math.sin(2 * math.pi * hour / 24.0)`
This allows the model to inherently understand Bengaluru's natural traffic cycles and apply dynamic penalties when events intersect with peak rush hours (8 AM - 11 AM, 5 PM - 9 PM).

### 2. Continuous Resource Optimization
Resource allocation replaces rigid heuristic step-logic with continuous algorithmic scaling based on the model's live Severity Index (0.0 to 1.0):
* **Manpower Scaling:** Officers deployed scale linearly with severity.
* **Physical Assets:** Barricade deployment scales exponentially based on critical severity thresholds, ensuring massive resources are only deployed for systemic threats.
* **Diversion Mapping:** Dynamically calculates perimeter radii up to 3.5km based on predicted event sprawl.

### 3. Interactive Simulation Mode
The frontend includes a developer testing suite("Interactive Simulation Mode") that allows dispatchers to bypass the ML baseline and stress-test the continuous optimization algorithms against extreme, non-historical anomalies.

## Repository Structure

```text
GRIDLOCK - P2/
│
├── backend/                 # FastAPI logic & Inference Engine
│   ├── main.py              # API Endpoints
│   └── inference.py         # Cyclical feature engineering & Model loading
│
├── frontend/                # User Interface
│   └── app.py               # Streamlit application
│
├── data/                    
│   ├── raw/                 # Original offline datasets
│   │   └── Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv
│   └── processed/           # Cleaned aggregation data for UI mapping
│       ├── events_cleaned.csv
│       └── zone_intelligence.csv
│
├── models/                  # Live deployment binaries (.pkl)
│   ├── astram_rf_model.pkl  # Structural Random Forest Regressor
│   ├── le_zone.pkl          # Zone Label Encoder
│   └── le_cause.pkl         # Event Cause Label Encoder
│
├── phase_1_archive/         # Quarantined Phase 1 heavy-ML experiments (Offline)
│   ├── experiments/         # Individual model training scripts
│   │   ├── 01_lightgbm.py
│   │   ├── 02_catboost.py
│   │   └── 03_xgboost.py
│   ├── scripts/             # Out-of-fold predictions and ensembling
│   │   ├── 01_oof_lightgbm.py
│   │   ├── 02_oof_catboost.py
│   │   └── 03_ensemble.py
│   └── README.md            # Notes on Phase 1 batch processing
│
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Installation & Execution

**1. Clone and Install Dependencies**

git clone <your-repo-link>
cd "Gridlock - P2"
pip install -r requirements.txt

**2. Boot the Intelligence Backend(Terminal 1)**
Ensure your terminal is in the root Gridlock - P2 directory, then start the FastAPI server:

uvicorn backend.main:app --reload
(You should see a confirmation log: ASTraM Random Forest Model Loaded Successfully.)

**3. Launch the Command Center UI(Terminal 2)**
Open a second terminal, navigate into the frontend directory, and start the Streamlit dashboard:

cd frontend
streamlit run app.py

The ASTraM Event Optimizer will automatically open in your browser at http://localhost:8501.