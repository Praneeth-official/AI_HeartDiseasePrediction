
# 🩺 AI-Based Heart Disease Risk Prediction Web App

A secure, interpretable machine learning system built to predict 10-year Coronary Heart Disease (CHD) risk using the Framingham Heart Study dataset. Designed as a clinical decision support tool for healthcare professionals and patients, the app integrates state-of-the-art ML models, SHAP-based explainability, and a user-friendly Dash interface.

---
## 📌 Project Overview

- 🔬 **Dataset**: Framingham Heart Study (4,240 patient records, 17 features)
- 🧪 **Models Used**: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost
- ⚖️ **Class Imbalance**: SMOTE applied
- 📊 **Model Evaluation**: Accuracy, Precision, Recall, F1-score, ROC-AUC
- 🔎 **Explainability**: SHAP (bar plots, beeswarm, decision and force plots)
- 🌐 **Web App**: Dash (Plotly) with login/signup/reset password
- 🔒 **Security**: CSV-based user authentication and session management

---

## 🧠 Methodology

### 🔹 Data Preprocessing
- Handled missing values
- Renamed columns for clarity
- Engineered new features:
  - `pulsePressure = systolic_bp - diastolic_bp`
  - `Smoking_Years = age - 21` (for current smokers)
- Encoded categorical variables (binary & ordinal)
- Scaled features with `StandardScaler`

### 🔹 Model Training & Evaluation
- Models trained with 80/20 split and SMOTE balancing
- Logistic Regression tuned with GridSearchCV:
  - `C=0.1`, `penalty=l2`, `solver=liblinear`
- Evaluation metrics focused on recall and F1-score due to healthcare context

### 🔹 Explainability with SHAP
- Visual insights with:
  - SHAP bar plot
  - Beeswarm plot
  - Decision plot
  - Force plot
- Highlights top predictors: Age, Smoking, BP, Cholesterol

---

## 📊 Results

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression| 0.65     | 0.24      | **0.61** | 0.35     | **0.6921** |
| Random Forest      | **0.80** | 0.32      | 0.28   | 0.30     | 0.6697  |
| XGBoost            | 0.80     | 0.27      | 0.18   | 0.22     | 0.6591  |

✅ **Best Recall & AUC**: Logistic Regression — chosen for clinical use  
📌 **Best Accuracy**: Random Forest — but less sensitive to CHD cases

---

## 💻 Web Application

Built with `Dash` and `Plotly`, the web app includes:
- User authentication (signup, login, reset password)
- Manual form entry for patient data
- Real-time prediction with risk percentage
- Visual explanation with SHAP plots

Accessible on: Desktop, Tablet, Mobile

---

## 📁 Folder Structure

```
AI_HeartDiseasePrediction/
├── app.py
├── requirements.txt
├── README.md
├── notebooks/
│   ├── setup.ipynb
│   ├── Distribution_*.png
│   └── catboost_info/
├── force_plot_patient5.html
└── ...
```

---

## 🚀 How to Run

```bash
git clone https://github.com/Praneeth-official/AI_HeartDiseasePrediction.git
cd AI_HeartDiseasePrediction
pip install -r requirements.txt
python app.py
```
Then visit `http://127.0.0.1:8055` in your browser.

---

## 🧪 Example Test Cases

| Scenario                     | Expected Risk | Reason |
|-----------------------------|---------------|--------|
| Young healthy female        | Very Low      | No risk factors |
| Elderly male smoker         | Very High     | Age, smoking, BP |
| Diabetic middle-aged male   | Medium        | Diabetes & BP |
| Young athlete               | Extremely Low | No risks |
| Stroke survivor             | High          | Stroke + hypertension |

---

## 🎯 Conclusion

- Logistic Regression is best suited for high-recall clinical screening
- SHAP enhances model transparency and trust
- Web app enables accessible and explainable risk prediction

---

## 📈 Future Work

- Integrate with Electronic Health Records (EHR)
- Deploy using Render or GCP
- Convert to Progressive Web App (PWA)
- Add API support and patient monitoring

---
## Screenshots
![Sign Up](https://github.com/Praneeth-official/AI_HeartDiseasePrediction/blob/master/screenshots/Signup.png?raw=true)
![Sign In](https://github.com/Praneeth-official/AI_HeartDiseasePrediction/blob/master/screenshots/Sign%20In.png?raw=true)
![Login](https://github.com/Praneeth-official/AI_HeartDiseasePrediction/blob/master/screenshots/Login.png)
![Dashboard](https://github.com/Praneeth-official/AI_HeartDiseasePrediction/blob/master/screenshots/Dashboard.png)
![User Input](https://github.com/Praneeth-official/AI_HeartDiseasePrediction/blob/master/screenshots/User%20Input.png)
![CHD Risk Score](https://github.com/Praneeth-official/AI_HeartDiseasePrediction/blob/master/screenshots/CHD%20Risk%20Score.png?raw=true)
![SHAP Plot](https://github.com/Praneeth-official/AI_HeartDiseasePrediction/blob/master/screenshots/SHAP%20Plot.png)
## 👨‍⚕️ Author

**Praneeth Badanapally**  
Master’s in Data Science, UMass Dartmouth  
📧 [badanapallypraneeth3@gmail.com](mailto:badanapallypraneeth3@gmail.com)  
🔗 [GitHub](https://github.com/Praneeth-official)
**Under the Guidance of Professor Dr.Iren Valova, PhD**
Associate Dean,
College of Engineering,
University of Massachusetts Dartmouth 
Contact
508-999-8502
iren.valova@umassd.edu
![School logo](https://upload.wikimedia.org/wikipedia/en/thumb/2/24/University_of_Massachusetts_Dartmouth_seal.svg/800px-University_of_Massachusetts_Dartmouth_seal.svg.png)
---

## 📄 License

This project is licensed under the MIT License.

---
