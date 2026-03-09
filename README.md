# 🔐 IEEE-CIS Fraud Detection System

A production-ready Machine Learning system to detect fraudulent financial transactions in real-time using a FastAPI REST API.

---

## 📌 Problem Statement

Credit card fraud is a major problem costing billions of dollars globally every year. Financial institutions need automated systems to detect fraudulent transactions in real-time before they are processed.

This project builds an end-to-end ML pipeline that:
- Analyzes transaction patterns
- Identifies suspicious behavior
- Returns fraud probability in real-time via REST API

---

## 📂 Dataset

- **Source:** [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection)
- **Size:** 590,540 transactions with 434 features
- **Files Used:** `train_transaction.csv` + `train_identity.csv`
- **Target:** `isFraud` (0 = Legitimate, 1 = Fraudulent)
- **Class Imbalance:** ~3.5% fraud transactions

---

## 🏗️ Project Architecture
```
Raw Data → Data Cleaning → Feature Engineering → Model Training → FastAPI Deployment
```

---

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| `Data_Understanding.ipynb` | EDA, data exploration, fraud distribution analysis |
| `Data_Cleaning.ipynb` | Missing value handling, feature removal, correlation analysis |
| `Feature_Engineering.ipynb` | Encoding, scaling, feature creation, SMOTE |
| `model_training.ipynb` | Model comparison, hyperparameter tuning, evaluation |

---

## 🔧 Data Preprocessing Pipeline

### 1. Data Cleaning
- Merged transaction + identity datasets on `TransactionID`
- Dropped columns with **>70% missing values**
- Removed low variance and identifier columns
- Handled **correlated features** (threshold > 0.7)
- Filled numerical missing values with **median**
- Filled categorical missing values with **"Unknown"**

### 2. Feature Engineering
- **Frequency Encoding** for high cardinality columns: `DeviceInfo`, `id_31`, `P_emaildomain`
- **Label Encoding** for categorical columns
- **Log Transformation** on `TransactionAmt` (right-skewed)
- **Decimal Feature:** `TransactionAmt_decimal` (fraud indicator)
- **Time Features:** `Transaction_hour`, `Transaction_day`
- **SMOTE** to handle class imbalance (3.5% → 50%)
- **StandardScaler** for numerical normalization

### 3. Class Imbalance Handling
```
Before SMOTE → Fraud: 1,995 | Legit: 71,953
After SMOTE  → Fraud: 71,953 | Legit: 71,953
```

---

## 🤖 Models Compared

| Model | Accuracy | Recall | F1 Score |
|-------|----------|--------|----------|
| Logistic Regression | ~85% | Low | Low |
| Decision Tree | ~95% | Medium | Medium |
| **Random Forest** | **99%** | **99%** | **99%** |
| Gradient Boosting | ~98% | High | High |
| XGBoost | ~98% | High | High |
| LightGBM | ~98% | High | High |

✅ **Final Model: Random Forest** with `RandomizedSearchCV` hyperparameter tuning

### Best Parameters
```python
{
  'n_estimators': 200,
  'max_depth': 20,
  'min_samples_split': 2,
  'min_samples_leaf': 1
}
```

### Model Performance
```
              precision    recall  f1-score
           0       0.99      1.00      0.99
           1       1.00      0.99      0.99
    accuracy                           0.99
```

---

## 🚀 API Deployment — FastAPI

### Tech Stack
- **Backend:** FastAPI + Uvicorn
- **ML:** Scikit-learn, Joblib
- **Data:** Pandas, NumPy
- **Validation:** Pydantic

### Project Structure
```
fraud-detection-api/
├── main.py              # FastAPI application
├── requirements.txt     # Dependencies
└── models/
    ├── fraud_model_top20.pkl    # Trained Random Forest model
    ├── scaler.pkl               # StandardScaler
    ├── freq_maps.pkl            # Frequency encoding maps
    └── feature_columns.pkl      # Feature column order
```

### How to Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
uvicorn main:app --reload

# Open API docs
http://127.0.0.1:8000/docs
```

### API Endpoints

#### GET `/`
Health check
```json
{"status": "Fraud Detection API is running!"}
```

#### POST `/predict`
Predict fraud probability

**Request:**
```json
{
  "TransactionAmt": 1500.0,
  "card2": 111.0,
  "card6": "credit",
  "P_emaildomain": "gmail.com",
  "C2": 15.0,
  "C4": 12.0,
  "C7": 10.0,
  "C8": 15.0,
  "C10": 8.0,
  "C11": 14.0,
  "C13": 20.0,
  "C14": 18.0,
  "M4": "M0",
  "M5": "F",
  "M6": "F",
  "V102": 0.0,
  "V280": 1.0,
  "V283": 0.0,
  "V294": 1.0,
  "DeviceInfo": "Unknown"
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

### Input Validation
- `TransactionAmt` must be positive
- `card6` must be `"credit"` or `"debit"`
- Invalid inputs return `422 Unprocessable Entity`

---

## 📊 Top 20 Important Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | C14 | 0.035 |
| 2 | C13 | 0.034 |
| 3 | card6 | 0.024 |
| 4 | V294 | 0.021 |
| 5 | M4 | 0.017 |
| 6 | C11 | 0.017 |
| 7 | C4 | 0.015 |
| 8 | C2 | 0.015 |
| 9 | P_emaildomain | 0.015 |
| 10 | TransactionAmt | 0.014 |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| ML Library | Scikit-learn |
| API Framework | FastAPI |
| Data Processing | Pandas, NumPy |
| Imbalance Handling | Imbalanced-learn (SMOTE) |
| Model Serialization | Joblib, Pickle |
| API Server | Uvicorn |

---

## 🔮 Future Improvements

| Feature | Description |
|---------|-------------|
| 🐳 Docker | Containerize the API for easy deployment |
| ☁️ Cloud Deploy | Deploy on AWS/GCP/Azure |
| 📊 MLflow | Model tracking and experiment logging |
| 🔄 Retraining | Automated model retraining pipeline |
| 🌐 Streamlit UI | Frontend dashboard for predictions |
| ⚡ XGBoost | Replace RF with XGBoost for better performance |
| 🔑 Auth | Add API key authentication |
| 📈 Logging | Request/response logging system |
| 🧪 Unit Tests | Add pytest test coverage |
| 📦 CI/CD | GitHub Actions for automated deployment |

---

## 📈 Model Improvement Plan

- Try **XGBoost/LightGBM** with full feature set
- **Feature Selection** using SHAP values
- **Threshold tuning** for better fraud recall
- **Real-time data pipeline** with Kafka
- **Graph Neural Networks** for transaction network analysis

---

## 👨‍💻 Author

**Shivam Nayak**  
AI/ML Engineer  
[GitHub](https://github.com/shivam-nayak-ds/fraud-detection-ml-system)
