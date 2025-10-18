import streamlit as st
import tensorflow as tf
import joblib
import numpy as np

# ✅ Load ANN model & scaler (from models/ folder inside project root)
model = tf.keras.models.load_model("models/ann_model.h5")
scaler = joblib.load("models/scaler.pkl")

st.title("🩺 AI-based Medical Diagnosis System")

# Example input fields (for Diabetes dataset)
preg = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 200, 120)
bp = st.number_input("Blood Pressure", 0, 150, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

if st.button("Predict"):
    # Create input array
    X = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    
    # Scale features using the saved scaler
    X = scaler.transform(X)
    
    # Make prediction
    prob = model.predict(X)[0][0]
    st.write(f"Prediction Probability: {prob:.2f}")
    
    if prob > 0.5:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

