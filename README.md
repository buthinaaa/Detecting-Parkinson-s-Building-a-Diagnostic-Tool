# 🧠 Parkinson's Disease Diagnostic Tool

A machine learning-powered diagnostic tool built using Python, XGBoost, and Streamlit to predict the likelihood of Parkinson's Disease based on medical and lifestyle features.

## 🚀 Features

* ✅ User-friendly web interface built with Streamlit
* 📊 Predicts Parkinson’s Disease risk using a trained XGBoost classifier
* 🔍 Collects key health indicators (age, lifestyle, symptoms, vitals)
* 🎯 Model optimized via GridSearchCV using recall score
* ⚙️ Handles imbalanced data through class weights

## 🏗️ Project Structure

```
📁 parkinsons-diagnosis-app
│
├── Detecting_Parkinson's_Building_a_Diagnostic_Tool.ipynb                                 
├── app.py                                     # Streamlit application
├── xgboost_parkinsons_model.pkl               # Trained XGBoost model
├── requirements.txt                           # Python dependencies
└── README.md                                  # Project documentation

```

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/parkinsons-diagnosis-app.git
cd parkinsons-diagnosis-app
```

2. **Create a virtual environment (optional)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## ▶️ Run the App

```bash
streamlit run app.py
```

The app will launch in your browser at `http://localhost:8501`.

## 📁 Model Details

* **Model Type:** XGBoost Classifier
* **Training Score Metric:** Recall
* **Hyperparameters Tuned:** `max_depth`, `n_estimators`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma`
* **Handling Imbalance:** Class weights set based on positive/negative sample ratios

## 📊 Input Features

| Feature Name                             | Description                                   |
| ---------------------------------------- | --------------------------------------------- |
| Age                                      | Age of the patient                            |
| Gender                                   | Male / Female                                 |
| Ethnicity                                | Encoded categories (e.g., White, Asian, etc.) |
| EducationLevel                           | Educational background                        |
| BMI                                      | Body Mass Index                               |
| Smoking / AlcoholConsumption             | Lifestyle habits                              |
| SleepQuality / DietQuality               | Wellness scores                               |
| FamilyHistoryParkinsons                  | Family history of PD                          |
| Medical History (Diabetes, Stroke, etc.) | Binary conditions                             |
| Symptoms (Tremor, Rigidity, etc.)        | Binary symptom indicators                     |
| Clinical Scores (UPDRS, MoCA)            | Assessment scores                             |

## 📌 Use Case

This tool is designed to assist medical professionals and researchers in:

* Early risk screening of Parkinson’s
* Health monitoring in clinical research
* Showcasing how ML can complement diagnostics (not replace doctors)

> ⚠️ **Disclaimer:** This tool is for research and educational purposes only. It is not a substitute for professional medical diagnosis.

## ✨ Future Improvements

* Add SHAP or LIME for explainability
* Deploy using Streamlit Cloud or Hugging Face Spaces
* Enable data logging and feedback

## 👩‍💻 Author

**Buthaina Esam**
*Machine Learning Engineer & AI Enthusiast*
