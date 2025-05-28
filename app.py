import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('xgboost_parkinsons_model.pkl')  

st.title("🧠 Parkinson's Diagnostic Tool")
st.write("Enter patient details to predict the likelihood of Parkinson's disease.")

binary_fields = {
    "Gender": ("Male", "Female"),
    "Smoking": ("No", "Yes"),
    "FamilyHistoryParkinsons": ("No", "Yes"),
    "TraumaticBrainInjury": ("No", "Yes"),
    "Hypertension": ("No", "Yes"),
    "Diabetes": ("No", "Yes"),
    "Depression": ("No", "Yes"),
    "Stroke": ("No", "Yes"),
    "Tremor": ("No", "Yes"),
    "Rigidity": ("No", "Yes"),
    "Bradykinesia": ("No", "Yes"),
    "PosturalInstability": ("No", "Yes"),
    "SpeechProblems": ("No", "Yes"),
    "SleepDisorders": ("No", "Yes"),
    "Constipation": ("No", "Yes"),
}

with st.form("patient_form"):
    age = st.slider("Age", 30, 100, 65)
    gender = st.selectbox("Gender", binary_fields["Gender"])
    ethnicity = st.selectbox("Ethnicity", ["White", "Black", "Asian", "Other"])
    education = st.selectbox("Education Level", ["None", "Primary", "Secondary", "Tertiary"])
    bmi = st.number_input("BMI", 10.0, 50.0, 25.5)
    smoking = st.selectbox("Smoking", binary_fields["Smoking"])
    alcohol = st.number_input("Alcohol Consumption", 0.0, 10.0, 1.0)
    physical = st.number_input("Physical Activity", 0.0, 10.0, 2.0)
    diet = st.number_input("Diet Quality", 0.0, 10.0, 3.0)
    sleep_quality = st.number_input("Sleep Quality", 0.0, 10.0, 4.0)
    family_history = st.selectbox("Family History of Parkinson's", binary_fields["FamilyHistoryParkinsons"])
    tbi = st.selectbox("Traumatic Brain Injury", binary_fields["TraumaticBrainInjury"])
    hypertension = st.selectbox("Hypertension", binary_fields["Hypertension"])
    diabetes = st.selectbox("Diabetes", binary_fields["Diabetes"])
    depression = st.selectbox("Depression", binary_fields["Depression"])
    stroke = st.selectbox("Stroke", binary_fields["Stroke"])
    systolic = st.number_input("Systolic BP", 90, 200, 135)
    diastolic = st.number_input("Diastolic BP", 60, 130, 85)
    chol_total = st.number_input("Cholesterol Total", 100.0, 400.0, 200.0)
    chol_ldl = st.number_input("Cholesterol LDL", 50.0, 300.0, 120.0)
    chol_hdl = st.number_input("Cholesterol HDL", 20.0, 100.0, 50.0)
    chol_trig = st.number_input("Cholesterol Triglycerides", 50.0, 500.0, 150.0)
    updrs = st.slider("UPDRS Score", 0.0, 100.0, 30.0)
    moca = st.slider("MoCA Score", 0.0, 30.0, 25.0)
    fa = st.number_input("Functional Assessment", 0.0, 10.0, 3.0)
    tremor = st.selectbox("Tremor", binary_fields["Tremor"])
    rigidity = st.selectbox("Rigidity", binary_fields["Rigidity"])
    brady = st.selectbox("Bradykinesia", binary_fields["Bradykinesia"])
    posture = st.selectbox("Postural Instability", binary_fields["PosturalInstability"])
    speech = st.selectbox("Speech Problems", binary_fields["SpeechProblems"])
    sleep_disorder = st.selectbox("Sleep Disorders", binary_fields["SleepDisorders"])
    constipation = st.selectbox("Constipation", binary_fields["Constipation"])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Convert categorical/binary fields to numeric values
    def bin_to_num(val): return 1 if val == "Yes" or val == "Female" else 0
    def encode_ethnicity(val): return {"White": 0, "Black": 1, "Asian": 2, "Other": 3}[val]
    def encode_education(val): return {"None": 0, "Primary": 1, "Secondary": 2, "Tertiary": 3}[val]

    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': bin_to_num(gender),
        'Ethnicity': encode_ethnicity(ethnicity),
        'EducationLevel': encode_education(education),
        'BMI': bmi,
        'Smoking': bin_to_num(smoking),
        'AlcoholConsumption': alcohol,
        'PhysicalActivity': physical,
        'DietQuality': diet,
        'SleepQuality': sleep_quality,
        'FamilyHistoryParkinsons': bin_to_num(family_history),
        'TraumaticBrainInjury': bin_to_num(tbi),
        'Hypertension': bin_to_num(hypertension),
        'Diabetes': bin_to_num(diabetes),
        'Depression': bin_to_num(depression),
        'Stroke': bin_to_num(stroke),
        'SystolicBP': systolic,
        'DiastolicBP': diastolic,
        'CholesterolTotal': chol_total,
        'CholesterolLDL': chol_ldl,
        'CholesterolHDL': chol_hdl,
        'CholesterolTriglycerides': chol_trig,
        'UPDRS': updrs,
        'MoCA': moca,
        'FunctionalAssessment': fa,
        'Tremor': bin_to_num(tremor),
        'Rigidity': bin_to_num(rigidity),
        'Bradykinesia': bin_to_num(brady),
        'PosturalInstability': bin_to_num(posture),
        'SpeechProblems': bin_to_num(speech),
        'SleepDisorders': bin_to_num(sleep_disorder),
        'Constipation': bin_to_num(constipation)
    }])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.markdown(f"### 🩺 Prediction: {'**Parkinson’s Detected**' if prediction == 1 else '**No Parkinson’s Detected**'}")
    st.markdown(f"#### 🧪 Probability of Parkinson’s: **{prob:.2%}**")
