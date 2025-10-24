import streamlit as st
import sqlite3
import hashlib
import os

# --- DATABASE SETUP ---
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
        return False  # Username already exists
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

# Initialize database
init_db()

# --- STREAMLIT APP ---
st.title("AI Medical Diagnosis System")

menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Register":
    st.subheader("Create a New Account")
    new_user = st.text_input("Username")
    new_password = st.text_input("Password", type='password')
    confirm_password = st.text_input("Confirm Password", type='password')
    
    if st.button("Register"):
        if new_password != confirm_password:
            st.error("Passwords do not match!")
        elif create_user(new_user, new_password):
            st.success("Account created successfully! You can now login.")
        else:
            st.error("Username already exists!")

elif choice == "Login":
    st.subheader("Login to Your Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    
    if st.button("Login"):
        if authenticate_user(username, password):
            st.success(f"Welcome {username}!")
            st.info("You can now use the AI Medical Diagnosis system.")

            # --- Call your AI diagnosis function here ---
            try:
                from AI_Medical_Diagnosis.ai_diagnosis_code import run_diagnosis
                run_diagnosis()  # replace with your function name
            except Exception as e:
                st.error("AI diagnosis system not found. Check import path.")
                st.error(str(e))
        else:
            st.error("Invalid username or password")
