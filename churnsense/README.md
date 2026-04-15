---
title: ChurnSense
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.36.0"
app_file: app/streamlit_app.py
pinned: false
---

# 🚀 ChurnSense — Customer Churn Prediction System

> Predict which customers are about to leave — before they actually leave.

---

## 🌐 Live Demo

👉 https://your-username-churnsense.hf.space

---

## 🧠 Problem Statement

Telecom and SaaS companies lose millions due to customer churn.

Most companies react **after** the customer leaves.
**ChurnSense predicts it before it happens.**

This allows businesses to:

* Reduce revenue loss 💸
* Target high-risk customers 🎯
* Improve retention strategies 📈

---

## ⚙️ What This Project Does

✔ Predicts churn probability
✔ Classifies customers into **Low / Medium / High risk**
✔ Explains *why* a customer may churn using SHAP
✔ Provides actionable insights

---

## 🧱 Tech Stack

* 🐍 Python
* 📊 pandas, numpy
* 🤖 scikit-learn, XGBoost, LightGBM
* 📉 SHAP (Explainability)
* 🌐 Streamlit (Frontend UI)
* ⚡ FastAPI (Backend API)
* 📦 MLflow (Experiment tracking)
* 🚀 HuggingFace Spaces (Deployment)
* 🗂 GitHub (Version control)

---

## 📊 Dataset

* IBM Telco Customer Churn Dataset
* 7,043 rows
* Binary classification (Churn: Yes / No)

---

## 🧪 Model Performance

| Model               | Accuracy | ROC-AUC |
| ------------------- | -------- | ------- |
| Logistic Regression | ~0.80    | ~0.84   |
| Random Forest       | ~0.82    | ~0.86   |
| XGBoost             | ~0.84    | ~0.88   |
| LightGBM ⭐          | ~0.85    | ~0.89   |

👉 **Best Model: LightGBM**

---

## 🔍 Key Insights (EDA)

* Customers with **month-to-month contracts** churn more
* **High monthly charges** → higher churn probability
* Lack of **tech support / online security** increases churn
* Long-term customers are more loyal

---

## 🧠 Explainability (SHAP)

The model doesn’t just predict — it explains:

* Which features increase churn risk
* Which features reduce churn risk
* Top 3 reasons for each prediction

---

## 🖥️ App Features

* 🎛 Interactive input form
* 📊 Churn probability display
* 🚨 Risk classification (Low / Medium / High)
* 🔎 Top churn reasons
* 💡 Actionable suggestions

---

## 📁 Project Structure

```
churnsense/
│
├── app/
│   ├── streamlit_app.py
│   └── api.py
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── encoder.pkl
│
├── data/
├── notebooks/
├── requirements.txt
└── README.md
```

---

## ⚡ How to Run Locally

```bash
git clone https://github.com/your-username/churnsense.git
cd churnsense
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## 🔌 API Usage (FastAPI)

Run API:

```bash
uvicorn app.api:app --reload
```

Test endpoint:

```bash
http://127.0.0.1:8000/docs
```

---

## 🚀 Deployment

Deployed using HuggingFace Spaces with Streamlit.

---

## 🧠 What I Learned

* Building end-to-end ML systems
* Handling real-world data issues
* Model explainability (SHAP)
* API development with FastAPI
* Deployment and production debugging

---

## 📸 Screenshots

(Add screenshots here after deployment)

---

## 📌 Future Improvements

* Real-time data integration
* Advanced feature engineering
* Model retraining pipeline
* Business dashboard

---

## 🤝 Connect

If you’re working on AI/ML or building something interesting, let’s connect.

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub
