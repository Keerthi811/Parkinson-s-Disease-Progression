# Predictive Analysis of Parkinson's Disease Progression Using Longitudinal Biomedical Voice Biomarkers

This is the repository for the Master's dissertation project: **"Predictive Analysis of Parkinson's Disease Progression Using Longitudinal Biomedical Voice Biomarkers"**.

The goal of this project is to model the progression of Parkinson's Disease (represented by metrics like the Unified Parkinson's Disease Rating Scale - UPDRS) using longitudinal, non-invasive voice recordings and acoustic features (voice biomarkers).

---

## Project Objectives

1. **Longitudinal Dynamics Modeling**: Capture the progression trajectory of motor and total UPDRS scores over time.
2. **Robust Voice Biomarker Extraction**: Clean, preprocess, and construct lag/rolling window features from biomedical voice biomarkers (Jitter, Shimmer, HNR, RPDE, DFA, PPE, etc.).
3. **Comparative Evaluation**: Build and evaluate classical machine learning models (Random Forest, XGBoost) and deep sequence models (LSTM/GRU) for progression prediction.
4. **Model Explainability**: Utilize SHAP (SHapley Additive exPlanations) to explain feature importance and identify which voice biomarkers are predictive of disease severity.

---

## Directory Structure

```
project_root/
│
├── data/
│   ├── raw/                 # Original, immutable raw datasets (e.g., parkinsons_updrs.csv)
│   └── processed/           # Processed datasets, ready for training
│
├── src/                     # Source code package
│   ├── data_validation/     # Scripts for validating raw data schema and integrity
│   ├── preprocessing/       # Audio/tabular cleaning, interpolation, outlier treatment
│   ├── feature_engineering/ # Temporal lagging, rolling statistics, trend extraction
│   ├── modeling/            # Classical ML model training and evaluation
│   ├── deep_learning/       # Neural networks, PyTorch model definitions, DL training
│   ├── explainability/      # Feature importance and SHAP analysis
│   ├── visualization/       # Utilities to generate plots, graphs, and trajectories
│   └── utils/               # Configurations, logger setups, and reproducibility seeds
│
├── models/                  # Serialized models (.pkl, .pt)
├── evaluation/              # Prediction CSVs, evaluation metric tables, model performance log
├── reports/                 # Artifacts for the dissertation
│   ├── figures/             # Output plots, feature importances, trajectory visualizations
│   └── tables/              # Result tables in CSV and LaTeX format
│
├── notebooks/               # Jupyter Notebooks for EDA and prototyping
├── app/                     # Dashboard application (e.g. Streamlit dashboard)
├── logs/                    # Execution log files
│
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation (this file)
├── config.yaml              # Centralized configuration settings
└── run_pipeline.py          # Central execution orchestrator script
```

---

## Getting Started

### 1. Setup Environment
Ensure Python 3.10+ is installed. Create and activate a virtual environment, then install requirements:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Placement
Place your dataset file inside the `data/raw/` directory. By default, the pipeline expects the file name `parkinsons_updrs.csv` (e.g., from the [UCI Parkinson's Telemonitoring Dataset](https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring)).

*If your dataset has a different name, adjust the `required_files` section inside `config.yaml` accordingly.*

### 3. Run Pipeline
To run the full end-to-end pipeline (validation, preprocessing, feature engineering, modeling, deep learning, explainability, visualization):

```bash
python run_pipeline.py
```

---

## Architectural Principles

- **Reproducibility**: Global seeds are applied across Python standard libraries, Numpy, and PyTorch (configured in `src/utils/reproducibility.py`).
- **Config-Driven**: No hardcoded variables. All hyperparameters, file paths, thresholds, and variables are read from `config.yaml` using a centralized loader.
- **Robust Exception Handling**: Custom exception definitions protect each step of the pipeline. In case of an failure, clear trace logs are preserved in `logs/pipeline.log`.
- **Modularity**: Individual pipeline stages are decoupled and can be run independently or imported elsewhere.
