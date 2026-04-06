# 🔐 Fraud Detection ML System (Work In Progress 🚧)

Production-grade machine learning pipeline for real-time credit card fraud detection.

## 📌 Project Status
Currently building the end-to-end pipeline. The project is in the **Data Engineering** phase.

### ✅ Completed Steps
- [x] Initial Project Structure setup.
- [x] High-performance Data Ingestion from CSV to SQLite Database.
- [x] Optimized Data Extraction using Polars.
- [x] Data Persistence (Saving fetched data as Parquet files).

### ⏳ To-Do (Upcoming)
- [ ] Data Preprocessing (Cleaning & Scaling).
- [ ] Feature Engineering (Log Transformation & Temporal Features).
- [ ] Model Training (Random Forest & XGBoost).
- [ ] MLflow Experiment Tracking.
- [ ] FastAPI Deployment.

## 🚀 How to Run (Current Stage)
1. **Setup Database:**
   ```bash
   python -m src.data_preprocessing
   ```
2. **Fetch and Save Data:**
   ```bash
   python -m src.data_fetch
   ```

---
*Created by Shivam Nayak*
