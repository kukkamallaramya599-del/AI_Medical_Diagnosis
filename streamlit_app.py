import streamlit as st
import requests
import numpy as np

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/predict"

st.title("🩺  Medical Diagnosis System")

st.write("Enter patient details to predict diabetes outcome:")

# Example features from Pima Indians dataset
features = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# Take user input
values = []
for f in features:
    val = st.number_input(f"Enter {f}:", min_value=0.0, step=0.1)
    values.append(val)

if st.button("Predict"):
    # Send to FastAPI
    try:
        response = requests.post(API_URL, json={"features": values})
        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ Probability: {result['probability']:.3f}")
            st.info(f"Predicted Outcome: {'Diabetic' if result['prediction']==1 else 'Non-Diabetic'}")
        else:
            st.error("❌ API error. Please check backend logs.")
    except Exception as e:
        st.error(f"❌ Connection error: {e}")
