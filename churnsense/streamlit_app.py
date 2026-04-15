import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

MODEL_FILES = ["best_model.pkl", "scaler.pkl", "encoder.pkl"]


@st.cache_resource
def load_models():
    repo_root = Path(__file__).resolve().parents[1]
    model_dir = repo_root / "models"

    if not model_dir.exists():
        raise FileNotFoundError(f"Models folder not found: {model_dir}")

    for filename in MODEL_FILES:
        if not (model_dir / filename).exists():
            raise FileNotFoundError(f"Missing model file: {model_dir / filename}")

    with open(model_dir / "best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(model_dir / "encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    return model, scaler, encoder


st.set_page_config(page_title="ChurnSense", layout="wide")
st.title("📊 ChurnSense — Customer Churn Prediction")
st.markdown("Predict which customers are likely to churn using the trained model.")

try:
    model, scaler, encoder = load_models()
except Exception as exc:
    st.error("Unable to load model artifacts.")
    st.write(str(exc))
    st.stop()


def user_input():
    return pd.DataFrame(
        [
            {
                "gender": st.sidebar.selectbox("Gender", ["Male", "Female"]),
                "SeniorCitizen": st.sidebar.selectbox("Senior Citizen", [0, 1]),
                "Partner": st.sidebar.selectbox("Partner", ["Yes", "No"]),
                "Dependents": st.sidebar.selectbox("Dependents", ["Yes", "No"]),
                "tenure": st.sidebar.slider("Tenure (months)", 0, 72, 12),
                "PhoneService": st.sidebar.selectbox("Phone Service", ["Yes", "No"]),
                "MultipleLines": st.sidebar.selectbox(
                    "Multiple Lines", ["Yes", "No", "No phone service"]
                ),
                "InternetService": st.sidebar.selectbox(
                    "Internet Service", ["DSL", "Fiber optic", "No"]
                ),
                "OnlineSecurity": st.sidebar.selectbox(
                    "Online Security", ["Yes", "No", "No internet service"]
                ),
                "OnlineBackup": st.sidebar.selectbox(
                    "Online Backup", ["Yes", "No", "No internet service"]
                ),
                "DeviceProtection": st.sidebar.selectbox(
                    "Device Protection", ["Yes", "No", "No internet service"]
                ),
                "TechSupport": st.sidebar.selectbox(
                    "Tech Support", ["Yes", "No", "No internet service"]
                ),
                "StreamingTV": st.sidebar.selectbox(
                    "Streaming TV", ["Yes", "No", "No internet service"]
                ),
                "StreamingMovies": st.sidebar.selectbox(
                    "Streaming Movies", ["Yes", "No", "No internet service"]
                ),
                "Contract": st.sidebar.selectbox(
                    "Contract", ["Month-to-month", "One year", "Two year"]
                ),
                "PaperlessBilling": st.sidebar.selectbox("Paperless Billing", ["Yes", "No"]),
                "PaymentMethod": st.sidebar.selectbox(
                    "Payment Method",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                ),
                "MonthlyCharges": st.sidebar.slider("Monthly Charges", 0.0, 150.0, 70.0),
                "TotalCharges": st.sidebar.slider("Total Charges", 0.0, 10000.0, 2000.0),
            }
        ]
    )


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    encoded = pd.DataFrame(
        encoder.transform(df[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index,
    )

    final_df = pd.concat(
        [df[numerical_cols].reset_index(drop=True), encoded.reset_index(drop=True)],
        axis=1,
    )
    final_df[numerical_cols] = scaler.transform(final_df[numerical_cols])

    return final_df


def get_top_features(model, X, top_n=3):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_.ravel())
    else:
        return None

    features = np.array(X.columns)
    top_idx = np.argsort(importances)[::-1][:top_n]

    return pd.DataFrame(
        {"Feature": features[top_idx], "Importance": importances[top_idx]}
    )


st.sidebar.header("Enter Customer Details")
input_df = user_input()

if st.button("🔍 Predict Churn"):
    processed_input = preprocess_input(input_df)

    prob = float(model.predict_proba(processed_input)[0, 1])
    prediction = int(model.predict(processed_input)[0])

    if prob > 0.7:
        risk = "🔴 High Risk"
    elif prob > 0.4:
        risk = "🟠 Medium Risk"
    else:
        risk = "🟢 Low Risk"

    st.subheader("Prediction Result")
    st.metric("Churn Probability", f"{prob:.2f}")
    st.write(f"Risk Level: {risk}")
    st.write("Predicted churn:", "Yes" if prediction == 1 else "No")

    feature_df = get_top_features(model, processed_input)
    if feature_df is not None:
        st.subheader("Top Model Features")
        st.table(feature_df)
    else:
        st.info("Feature importance is not available for this model.")