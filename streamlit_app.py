# streamlit_app.py
import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import os
import sqlite3
import hashlib

# ---------------------------
# 0. DATABASE SETUP
# ---------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                  (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    data = c.fetchone()
    conn.close()
    if data and data[0] == hash_password(password):
        return True
    return False

init_db()

# ---------------------------
# 1. Session State for Login
# ---------------------------
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""

def logout():
    st.session_state['logged_in'] = False
    st.session_state['username'] = ""
    st.success("Logged out successfully")

# ---------------------------
# 2. Load Models and Scalers
# ---------------------------
model_dir = "models"

def load_model_safe(model_path):
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.warning(f"Model file not found: {model_path}")
        return None

def load_scaler_safe(scaler_path):
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    else:
        st.warning(f"Scaler file not found: {scaler_path}")
        return None

diabetes_model = load_model_safe(os.path.join(model_dir, "diabetes_model.h5"))
diabetes_scaler = load_scaler_safe(os.path.join(model_dir, "diabetes_scaler.pkl"))

heart_model = load_model_safe(os.path.join(model_dir, "heart_model.h5"))
heart_scaler = load_scaler_safe(os.path.join(model_dir, "heart_scaler.pkl"))

kidney_model = load_model_safe(os.path.join(model_dir, "kidney_model.h5"))
kidney_scaler = load_scaler_safe(os.path.join(model_dir, "kidney_scaler.pkl"))

# ---------------------------
# 3. Feature Names
# ---------------------------
DIABETES_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

KIDNEY_FEATURES = [
    "Age", "BloodPressure", "SpecificGravity", "Albumin", "Sugar", "Bacteria"
]

HEART_KEY_FEATURES = {
    0: "Age",
    1: "Sex (1=Male,0=Female)",
    3: "RestingBP",
    4: "Cholesterol"
}

# ---------------------------
# 4. App UI
# ---------------------------
st.title("🩺 AI-based Medical Diagnosis System")

menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

# ---------- REGISTER PAGE ----------
if choice == "Register":
    st.subheader("Create a New Account")
    new_user = st.text_input("Username")
    new_password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    if st.button("Register"):
        if new_password != confirm_password:
            st.error("Passwords do not match!")
        elif create_user(new_user, new_password):
            st.success("Account created successfully! You can now login.")
        else:
            st.error("Username already exists!")

# ---------- LOGIN PAGE ----------
elif choice == "Login":
    st.subheader("Login to Your Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success(f"Logged in as {username}")
        else:
            st.error("Invalid username or password")

# ---------- PREDICTION PAGE ----------
if st.session_state['logged_in']:
    st.write(f"Welcome, Dr. {st.session_state['username']}!")
    if st.button("Logout"):
        logout()

    st.subheader("Choose Disease Prediction")
    disease = st.selectbox("Select Disease", ["Diabetes", "Heart Disease", "Kidney Disease"])

    # --------------------------- Diabetes Prediction ---------------------------
    if disease == "Diabetes" and diabetes_model and diabetes_scaler:
        with st.form("diabetes_form"):
            st.write("Enter patient details for Diabetes prediction:")
            inputs_list = []
            for fname in DIABETES_FEATURES:
                val = st.number_input(fname, value=0.0)
                inputs_list.append(val)
            submit = st.form_submit_button("Predict Diabetes")
            if submit:
                inputs = np.array([inputs_list])
                inputs_scaled = diabetes_scaler.transform(inputs)
                prediction_raw = diabetes_model.predict(inputs_scaled)
                prediction_label = "Positive" if prediction_raw[0][0] >= 0.5 else "Negative"
                st.success(f"Diabetes Prediction: {prediction_label}")

    # --------------------------- Kidney Disease Prediction ---------------------------
    elif disease == "Kidney Disease" and kidney_model and kidney_scaler:
        total_features = kidney_scaler.n_features_in_
        with st.form("kidney_form"):
            st.write("Enter patient details for Kidney Disease prediction:")
            inputs_array = np.zeros(total_features)
            key_features_idx = {0:"Age",1:"BloodPressure",2:"SpecificGravity",3:"Albumin",4:"Sugar",5:"Bacteria"}
            for idx, fname in key_features_idx.items():
                val = st.number_input(fname, value=0.0)
                inputs_array[idx] = val
            submit = st.form_submit_button("Predict Kidney Disease")
            if submit:
                inputs_array = inputs_array.reshape(1, -1)
                inputs_scaled = kidney_scaler.transform(inputs_array)
                prediction_raw = kidney_model.predict(inputs_scaled)
                prediction_label = "Positive" if prediction_raw[0][0] >= 0.5 else "Negative"
                st.success(f"Kidney Disease Prediction: {prediction_label}")

    # --------------------------- Heart Disease Prediction ---------------------------
    elif disease == "Heart Disease" and heart_model and heart_scaler:
        total_features = heart_scaler.n_features_in_
        with st.form("heart_form"):
            st.write("Enter key patient details for Heart Disease prediction:")
            inputs_array = np.zeros(total_features)
            for idx, fname in HEART_KEY_FEATURES.items():
                val = st.number_input(fname, value=0.0)
                inputs_array[idx] = val
            submit = st.form_submit_button("Predict Heart Disease")
            if submit:
                inputs_array = inputs_array.reshape(1, -1)
                inputs_scaled = heart_scaler.transform(inputs_array)
                prediction_raw = heart_model.predict(inputs_scaled)
                prediction_label = "Positive" if prediction_raw[0][0] >= 0.5 else "Negative"
                st.success(f"Heart Disease Prediction: {prediction_label}")
