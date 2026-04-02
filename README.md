# 🔐 Fraud-Shield: End-to-End Financial Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn)
![Docker](https://img.shields.io/badge/Deployment-Docker_Ready-2496ED?style=for-the-badge&logo=docker)

### 🚀 Business Impact
In the financial sector, detecting fraud in milliseconds is the difference between saving or losing millions. This project implements a **Real-Time Fraud Scoring Engine** that processes transaction metadata and identity features to provide instantaneous risk probability.



---

### 🛠️ Advanced Engineering Highlights
* **Imbalanced Data Handling:** Leveraged **SMOTE** (Synthetic Minority Over-sampling Technique) to solve the 3.5% fraud class imbalance, ensuring high recall for fraudulent cases.
* **Feature Engineering:** Implemented **Frequency Encoding** for high-cardinality data and custom time-based features to capture suspicious temporal patterns.
* **Production-Grade API:** Built with **FastAPI** including Pydantic data validation to ensure the model never receives "garbage" data.
* **Modular Pipeline:** Separate scripts for training, feature engineering, and inference for better maintainability.



---

### 📊 Performance Summary (Random Forest)
| Metric | Score |
| :--- | :--- |
| **Accuracy** | 99% |
| **Precision (Fraud)** | 1.00 |
| **Recall (Fraud)** | 0.99 |
| **F1-Score** | 0.99 |

---

### 📡 API Live Demo (Local)
Once the server is running, you can interact with the model via Swagger UI:
`http://127.0.0.1:8000/docs`



---

### 🏗️ Future Scalability (Roadmap)
- [ ] **Dockerization:** Wrap the API in a Docker container for cloud-agnostic deployment.
- [ ] **MLflow Integration:** To track model versions and hyperparameter experiments.
- [ ] **XGBoost Optimization:** Compare latency vs accuracy against the current Random Forest model.
