import joblib
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator

app = FastAPI(title="Fraud Detection API")

model = joblib.load("models/fraud_model_top20.pkl")
scaler = pickle.load(open("models/scaler.pkl", "rb"))
freq_maps = pickle.load(open("models/freq_maps.pkl", "rb"))

top_features = ['C14','C13','card6','V294','M4','C11','C4','C2',
                'P_emaildomain','TransactionAmt','C10','V102','card2',
                'C8','M6','M5','TransactionAmt_Log','V283','C7','V280']

class TransactionInput(BaseModel):
    TransactionAmt: float
    card2: float
    card6: str
    P_emaildomain: str
    C2: float = 0
    C4: float = 0
    C7: float = 0
    C8: float = 0
    C10: float = 0
    C11: float = 0
    C13: float = 0
    C14: float = 0
    M4: str = "M0"
    M5: str = "F"
    M6: str = "F"
    V102: float = 0
    V280: float = 0
    V283: float = 0
    V294: float = 0
    DeviceInfo: str = "Unknown"

    @validator('TransactionAmt')
    def amt_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("TransactionAmt must be positive")
        return v

    @validator('card6')
    def card6_valid(cls, v):
        if v not in ["credit", "debit"]:
            raise ValueError("card6 must be credit or debit")
        return v

@app.get("/")
def home():
    return {"status": "Fraud Detection API is running!"}

@app.post("/predict")
def predict(data: TransactionInput):
    try:
        input_dict = data.dict()

        # Frequency encoding
        for col in ['DeviceInfo', 'P_emaildomain']:
            if col in input_dict and col in freq_maps:
                input_dict[col] = freq_maps[col].get(input_dict[col], 0)

        # card6 encoding
        input_dict['card6'] = 1 if input_dict['card6'] == 'credit' else 2

        # M columns encoding
        m4_map = {"M0": 0, "M1": 1, "M2": 2, "M3": 3}
        input_dict['M4'] = m4_map.get(input_dict['M4'], 0)
        input_dict['M5'] = 1 if input_dict['M5'] == 'T' else 0
        input_dict['M6'] = 1 if input_dict['M6'] == 'T' else 0

        # TransactionAmt_Log feature
        input_dict['TransactionAmt_Log'] = np.log1p(input_dict['TransactionAmt'])

        # DataFrame banao
        input_df = pd.DataFrame([input_dict])

        # Sirf top features lo
        input_df = input_df[top_features]

        # Predict - naya model already scaled data pe train hua hai
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return {
            "prediction": int(prediction),
            "fraud_probability": round(float(probability), 4),
            "result": "FRAUD" if prediction == 1 else "NOT FRAUD"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))