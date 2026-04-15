# app/streamlit_app.py
import shap
shap.initjs()
import streamlit as st
import pandas as pd
import numpy as np
import pickle

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
encoder_path = os.path.join(BASE_DIR, "models", "encoder.pkl")
# -----------------------------
# LOAD MODELS
# -----------------------------
# app/streamlit_app.py

from pathlib import Path
import pickle
import streamlit as st

@st.cache_resource
def load_models():
    BASE_DIR = Path(__file__).resolve().parents[1]
    MODEL_DIR = BASE_DIR / "models"

    with open(MODEL_DIR / "best_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open(MODEL_DIR / "encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    return model, scaler, encoder

model, scaler, encoder = load_models()

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="ChurnSense", layout="wide")

st.title("📊 ChurnSense — Customer Churn Prediction")
st.markdown("Predict which customers are likely to churn and understand why.")

# -----------------------------
# INPUT FORM
# -----------------------------
st.sidebar.header("Enter Customer Details")

def user_input():
    data = {
        "gender": st.sidebar.selectbox("Gender", ["Male", "Female"]),
        "SeniorCitizen": st.sidebar.selectbox("Senior Citizen", [0, 1]),
        "Partner": st.sidebar.selectbox("Partner", ["Yes", "No"]),
        "Dependents": st.sidebar.selectbox("Dependents", ["Yes", "No"]),
        "tenure": st.sidebar.slider("Tenure (months)", 0, 72, 12),
        "PhoneService": st.sidebar.selectbox("Phone Service", ["Yes", "No"]),
        "MultipleLines": st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"]),
        "InternetService": st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]),
        "OnlineSecurity": st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"]),
        "OnlineBackup": st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"]),
        "DeviceProtection": st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"]),
        "TechSupport": st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"]),
        "StreamingTV": st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"]),
        "StreamingMovies": st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"]),
        "Contract": st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
        "PaperlessBilling": st.sidebar.selectbox("Paperless Billing", ["Yes", "No"]),
        "PaymentMethod": st.sidebar.selectbox("Payment Method", 
                                             ["Electronic check", "Mailed check", 
                                              "Bank transfer (automatic)", 
                                              "Credit card (automatic)"]),
        "MonthlyCharges": st.sidebar.slider("Monthly Charges", 0.0, 150.0, 70.0),
        "TotalCharges": st.sidebar.slider("Total Charges", 0.0, 10000.0, 2000.0)
    }
    return pd.DataFrame([data])

input_df = user_input()

# -----------------------------
# PREPROCESS INPUT
# -----------------------------
def preprocess_input(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=np.number).columns

    # Encode categorical
    encoded = pd.DataFrame(
        encoder.transform(df[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    # Combine
    final_df = pd.concat([df[numerical_cols].reset_index(drop=True), encoded], axis=1)

    # Scale numerical
    final_df[numerical_cols] = scaler.transform(final_df[numerical_cols])

    return final_df

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔍 Predict Churn"):

    processed_input = preprocess_input(input_df)

    prob = model.predict_proba(processed_input)[0][1]
    prediction = model.predict(processed_input)[0]

    # Risk label
    if prob > 0.7:
        risk = "🔴 High Risk"
    elif prob > 0.4:
        risk = "🟠 Medium Risk"
    else:
        risk = "🟢 Low Risk"

    st.subheader("Prediction Result")
    st.metric("Churn Probability", f"{prob:.2f}")
    st.write(f"Risk Level: {risk}")

    # -----------------------------
    # SHAP EXPLANATION
    # -----------------------------
    explainer = shap.Explainer(model, processed_input)
    shap_values = explainer(processed_input)

    shap_vals = shap_values[0].values
    feature_names = processed_input.columns

    impact_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": shap_vals
    })

    impact_df["AbsImpact"] = np.abs(impact_df["Impact"])
    top_features = impact_df.sort_values(by="AbsImpact", ascending=False).head(3)

    st.subheader("Top Reasons for Churn Risk")
    st.table(top_features[["Feature", "Impact"]])

    st.info("These factors contributed most to the churn prediction.")