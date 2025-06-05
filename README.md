# 🩺 AI-Based Heart Disease Risk Prediction Web App

This project predicts the **10-year risk of Coronary Heart Disease (CHD)** using machine learning models trained on the **Framingham Heart Study** dataset. It features a **Dash web application** with secure login, patient form input, model predictions, SHAP-based feature explanations, and a doctor-patient friendly interface.

---

## 🚀 Features

- 🧠 ML Models: Logistic Regression, Random Forest, XGBoost
- 📊 SHAP Explainability: Bar, beeswarm, and force plots
- 🔐 Secure Login, Signup, and Password Reset
- 👨‍⚕️ Manual Patient Form Input (with age, BP, cholesterol, etc.)
- 📈 Risk Prediction & Visual Interpretation
- 📤 (Optional) Downloadable PDF report with results

---

## 📂 Project Structure

AI_HeartDiseasePrediction/
├── data/
│ └── processed/processed_framingham_heart_study.csv
├── models/
│ └── logistic_regression_model.pkl
| └── random_forest_model.pkl
| └── xgboost_model.pkl
├── notebooks/
│ └── setup.ipynb
├── app.py
├── users.csv
├── requirements.txt
└── README.md

## How to install:
pip install -r requirements.txt
