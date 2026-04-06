<div align="center">

# 🔐 Fraud Detection ML System

**Production-grade machine learning pipeline for real-time credit card fraud detection**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.18-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=github-actions&logoColor=white)](https://github.com/features/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [API Docs](#-api-endpoints) · [Results](#-model-performance)

</div>

---

## 📌 Problem Statement

Credit card fraud costs the global economy **$32+ billion annually**. Financial institutions need automated, real-time detection systems to identify fraudulent transactions before they are processed.

This project builds an **end-to-end ML pipeline** that:
- Ingests and preprocesses 590K+ transactions with 434 features
- Handles extreme class imbalance (3.5% fraud) using SMOTE
- Trains and evaluates multiple ML models with MLflow experiment tracking
- Serves predictions via a production REST API with <100ms latency
- Provides a real-time monitoring dashboard via Streamlit

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔄 **Modular Pipeline** | Separate preprocessing → feature engineering → training → evaluation stages |
| 📊 **MLflow Tracking** | Full experiment tracking with params, metrics, and model artifacts |
| ⚡ **FastAPI Serving** | REST API with Pydantic validation and Swagger docs |
| 🖥️ **Streamlit Dashboard** | Interactive UI for real-time fraud prediction |
| 🐳 **Docker Ready** | Multi-service containerization (API + UI + MLflow) |
| 🧪 **Tested** | Pytest suite for API endpoints and preprocessing functions |
| 🔁 **CI/CD** | GitHub Actions pipeline with lint → test → Docker build |
| 📝 **Logging** | Rotating file-based logging across all modules |

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Raw Data   │────▶│  Preprocessing   │────▶│    Feature Eng   │────▶│   Training   │
│  (590K txn) │     │  Clean + Encode  │     │  Select Top 50   │     │  RandomForest│
└─────────────┘     └──────────────────┘     └─────────────────┘     └──────┬───────┘
                                                                            │
                    ┌──────────────────┐     ┌─────────────────┐            │
                    │  Streamlit UI    │◀───▶│   FastAPI        │◀───────────┘
                    │  Dashboard       │     │   /predict       │     ┌──────────────┐
                    └──────────────────┘     └─────────────────┘     │   MLflow      │
                                                                     │   Tracking    │
                                                                     └──────────────┘
```

---

## 📂 Project Structure

```
fraud-detection-system/
├── .github/workflows/
│   └── ci.yml                       # CI/CD pipeline
├── data/
│   ├── raw/                         # Original CSVs (gitignored)
│   ├── processed/                   # Pickle files (gitignored)
│   └── dataset_link.txt             # Kaggle download link
├── src/
│   ├── config.py                    # Central configuration
│   ├── logger.py                    # Rotating file logger
│   ├── data_preprocessing.py        # Clean, encode, scale, SMOTE
│   ├── feature_engineering.py       # Log transform, time features, top-N
│   ├── train.py                     # RF training + MLflow logging
│   ├── evaluate.py                  # Metrics, plots, MLflow artifacts
│   └── utils.py                     # Helper functions
├── fraud-detection-api/
│   ├── main.py                      # FastAPI application
│   ├── streamlit_app.py             # Streamlit frontend
│   └── models/                      # Serialized model artifacts
├── notebooks/                       # Jupyter notebooks (EDA → Training)
├── tests/
│   ├── test_api.py                  # API endpoint tests
│   └── test_preprocessing.py        # Data pipeline tests
├── reports/                         # Generated plots & metrics
├── models/                          # Trained model artifacts
├── Dockerfile                       # API container
├── Dockerfile.streamlit             # UI container
├── docker-compose.yml               # Multi-service orchestration
├── run_pipeline.py                  # Pipeline orchestrator (CLI)
├── requirements.txt                 # Pinned dependencies
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [Kaggle IEEE-CIS Dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

### 1. Clone & Install

```bash
git clone https://github.com/shivam-nayak-ds/fraud-detection-ml-system.git
cd fraud-detection-ml-system
pip install -r requirements.txt
```

### 2. Download Dataset

Place these files in `data/raw/`:
- `train_transaction.csv`
- `train_identity.csv`

### 3. Run the Pipeline

```bash
# Full pipeline (preprocess → train → evaluate)
python run_pipeline.py

# Or run individual stages
python run_pipeline.py --stage preprocess
python run_pipeline.py --stage train
python run_pipeline.py --stage evaluate
```

### 4. Launch the API

```bash
cd fraud-detection-api
uvicorn main:app --reload
# API docs: http://127.0.0.1:8000/docs
```

### 5. Launch the Dashboard

```bash
streamlit run fraud-detection-api/streamlit_app.py
# UI: http://localhost:8501
```

### 6. Docker (Optional)

```bash
docker compose up --build
# API:     http://localhost:8000
# UI:      http://localhost:8501
# MLflow:  http://localhost:5000
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection) |
| **Transactions** | 590,540 |
| **Features** | 434 (transaction + identity) |
| **Target** | `isFraud` (0 = Legit, 1 = Fraud) |
| **Imbalance** | ~3.5% fraud rate |

---

## 🔧 Data Pipeline

### Preprocessing
- **Merge** transaction + identity datasets on `TransactionID`
- **Drop** columns with >50% missing values
- **Impute** numerical → median, categorical → mode
- **Encode** low cardinality → Label, high cardinality → Frequency
- **Scale** all numerics with StandardScaler
- **SMOTE** to balance classes (3.5% → 50%)

### Feature Engineering
- **Log transform** on `TransactionAmt` (right-skewed)
- **Decimal feature**: `TransactionAmt_decimal` (fraud indicator)
- **Time features**: `Transaction_hour`, `Transaction_day`
- **Feature selection**: Top 50 by Random Forest importance

---

## 🤖 Model Performance

| Model | Accuracy | Recall | F1 Score | ROC-AUC |
|-------|----------|--------|----------|---------|
| Logistic Regression | ~85% | Low | Low | — |
| Decision Tree | ~95% | Medium | Medium | — |
| **Random Forest** | **99%** | **99%** | **99%** | **0.99** |
| XGBoost | ~98% | High | High | — |
| LightGBM | ~98% | High | High | — |

**Final Model: Random Forest** with `RandomizedSearchCV`

```
              precision    recall  f1-score
         0       0.99      1.00      0.99
         1       1.00      0.99      0.99
  accuracy                           0.99
```

### Best Hyperparameters
```python
{
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 2,
    "min_samples_leaf": 1
}
```

---

## 🌐 API Endpoints

### `GET /` — Health Check
```json
{"status": "Fraud Detection API is running!"}
```

### `POST /predict` — Fraud Prediction

**Request:**
```json
{
    "TransactionAmt": 1500.0,
    "card2": 111.0,
    "card6": "credit",
    "P_emaildomain": "gmail.com",
    "C2": 15.0, "C4": 12.0, "C7": 10.0, "C8": 15.0,
    "C10": 8.0, "C11": 14.0, "C13": 20.0, "C14": 18.0,
    "M4": "M0", "M5": "F", "M6": "F",
    "V102": 0.0, "V280": 1.0, "V283": 0.0, "V294": 1.0,
    "DeviceInfo": "Windows"
}
```

**Response:**
```json
{
    "prediction": 1,
    "fraud_probability": 0.87,
    "result": "FRAUD"
}
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.11 |
| **ML** | Scikit-learn, Imbalanced-learn (SMOTE) |
| **Experiment Tracking** | MLflow |
| **API** | FastAPI + Uvicorn + Pydantic |
| **Frontend** | Streamlit |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Containerization** | Docker + Docker Compose |
| **CI/CD** | GitHub Actions |
| **Testing** | Pytest |

---

## 📓 Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `Data_Understanding.ipynb` | EDA, fraud distribution, feature analysis |
| 2 | `Data_Cleaning_.ipynb` | Missing values, correlations, outliers |
| 3 | `Feature_Engineering (1).ipynb` | Encoding, scaling, SMOTE, feature creation |
| 4 | `model_training.ipynb` | Model comparison, hyperparameter tuning |

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v
pytest tests/test_preprocessing.py -v
```

---

## 👨‍💻 Author

**Shivam Nayak**
AI/ML Engineer
[GitHub](https://github.com/shivam-nayak-ds)

---

<div align="center">

⭐ **Star this repo if you found it useful!** ⭐

</div>
