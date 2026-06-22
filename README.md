# ASTraM Event Optimizer
**Flipkart Gridlock Hackathon 2.0 - Phase 2 Submission**

The ASTraM Event Optimizer is a real-time, machine-learning-driven command center designed to shift municipal traffic management from *reactive* to *proactive*. By predicting the spatiotemporal impact of localized events(e.g., waterlogging, public events) before they escalate, ASTraM instantly generates a deterministic Dispatch Manifest to optimize police manpower and physical resource deployment.

## System Architecture & ML Strategy

To ensure maximum scalability and performance, this project utilizes a fully decoupled architecture:

* **Frontend(Streamlit + PyDeck):** Handles real-time 3D geospatial rendering, user inputs and metric visualization.
* **Backend API(FastAPI + Uvicorn):** A lightning-fast, asynchronous bridge that houses the live intelligence engine.
* **Machine Learning Pipeline(Scikit-Learn):** * **Phase 1 (Archive):** Our initial data science heavy-lifting involved training complex offline ensembles(LightGBM/CatBoost) to establish baseline predictions and feature importance. These batch-processing notebooks are preserved in the `archive_phase1/` folder.
 * **Phase 2 (Live Inference):** For real-time API deployment, we transitioned to a lightweight, structural **Random Forest Regressor**. This live model dynamically calculates severity scores on the fly rather than relying on static historical lookups.

## Core Intelligence Features

### 1. The Dispatch Policy Engine
Unlike naive ML implementations, we do not let a black-box model dictate police deployment. Our ML engine strictly predicts the **Severity Probability**. Downstream resource allocation is handled by a deterministic, bounded *Dispatch Policy Engine*, ensuring highly interpretable, mathematically consistent deployment instructions that city officials can manually tune.

### 2. Cyclical Feature Engineering
The engine does not treat time as a static integer. Datetime inputs are mathematically transformed into continuous trigonometric sine and cosine waves:
`hour_sin = math.sin(2 * math.pi * hour / 24.0)`
This allows the model to inherently understand Bengaluru's natural traffic cycles and apply dynamic penalties when events intersect with peak rush hours (8 AM - 11 AM, 5 PM - 9 PM).

### 3. Historical System Dynamics
To accurately measure ASTraM's proactive impact, the backend cross-references the ML's proactive clearance time against Historical Baseline Averages for specific event causes(e.g., Unmanaged Waterlogging = 240 mins), mapping the exact clearance time saved by intercepting the cascading gridlock queue.

### 4. Interactive Simulation Mode
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

git clone <https://github.com/10Aryan07/Flipkart_Gridlock_Hackathon_2.0_Astram-Event-Optimizer.git>

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