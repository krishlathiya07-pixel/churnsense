# app/api.py

from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import shap

# -----------------------------
# LOAD MODELS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

with open(MODEL_DIR / "best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(MODEL_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open(MODEL_DIR / "encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# -----------------------------
# FASTAPI INIT
# -----------------------------
app = FastAPI(title="ChurnSense API")

# -----------------------------
# INPUT SCHEMA
# -----------------------------
class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# -----------------------------
# PREPROCESS FUNCTION
# -----------------------------
def preprocess_input(data: Customer):
    df = pd.DataFrame([data.dict()])

    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=np.number).columns

    encoded = pd.DataFrame(
        encoder.transform(df[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    final_df = pd.concat([df[numerical_cols], encoded], axis=1)
    final_df[numerical_cols] = scaler.transform(final_df[numerical_cols])

    return final_df

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/health")
def health():
    return {"status": "API running"}

# -----------------------------
# PREDICT ENDPOINT
# -----------------------------
@app.post("/predict")
def predict(data: Customer):

    processed = preprocess_input(data)

    prob = model.predict_proba(processed)[0][1]
    prediction = model.predict(processed)[0]

    # Risk level
    if prob > 0.7:
        risk = "High"
    elif prob > 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    # SHAP
    explainer = shap.Explainer(model, processed)
    shap_values = explainer(processed)

    shap_vals = shap_values[0].values
    feature_names = processed.columns

    impact_df = pd.DataFrame({
        "feature": feature_names,
        "impact": shap_vals
    })

    impact_df["abs"] = np.abs(impact_df["impact"])
    top_features = impact_df.sort_values(by="abs", ascending=False).head(3)

    return {
        "churn_probability": round(float(prob), 3),
        "prediction": int(prediction),
        "risk_level": risk,
        "top_reasons": top_features[["feature", "impact"]].to_dict(orient="records")
    }
    
    