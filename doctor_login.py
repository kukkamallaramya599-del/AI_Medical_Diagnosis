# ---------------------------
# Generic Login with Password
# ---------------------------
import streamlit as st

st.title("🩺 AI Medical Diagnosis System")

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""

# Show login form if not logged in
if not st.session_state['logged_in']:
    username = st.text_input("Enter your name")
    password = st.text_input("Enter a password", type="password")
    
    if st.button("Login"):
        if username.strip() != "" and password.strip() != "":
            # Save in session state
            st.session_state['logged_in'] = True
            st.session_state['username'] = username.strip()
            st.session_state['password'] = password.strip()
            st.success(f"Welcome Dr. {st.session_state['username']}!")
        else:
            st.error("❌ Please enter both a name and a password")
    
    st.stop()  # Stop the rest of the app until login
